from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import torch
import yaml
import pyloudnorm as pyln

from scripts import enhance as enhancer
from scripts import audio_utils


def _resolve_device(configured: str | None = None) -> str:
    if configured and configured != "auto":
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"


def _beta_wiener_mask(
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    beta: float = 1.1,
    floor: float = 0.05,
    fft_size: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Apply a soft mask derived from the accompaniment stem."""
    if beta <= 0:
        return vocals
    ch_axis = 0 if vocals.ndim == 1 else 0
    enhanced = []
    if vocals.ndim == 1:
        vocals = vocals[np.newaxis, :]
    if accompaniment.ndim == 1:
        accompaniment = accompaniment[np.newaxis, :]
    for ch in range(vocals.shape[0]):
        v = vocals[ch]
        if ch < accompaniment.shape[0]:
            a = accompaniment[ch]
        else:
            a = accompaniment[0]
        spec_v = librosa.stft(v, n_fft=fft_size, hop_length=hop_length)
        spec_a = librosa.stft(a, n_fft=fft_size, hop_length=hop_length)
        mag_v = np.abs(spec_v)
        mag_a = np.abs(spec_a)
        mask = mag_v**beta / (mag_v**beta + mag_a**beta + 1e-12)
        mask = np.clip(floor + (1.0 - floor) * mask, floor, 1.0)
        enhanced_spec = spec_v * mask
        enhanced_signal = librosa.istft(
            enhanced_spec, hop_length=hop_length, length=v.shape[0]
        )
        enhanced.append(enhanced_signal.astype(np.float32))
    enhanced_arr = np.stack(enhanced, axis=0)
    return enhanced_arr if ch_axis == 0 else enhanced_arr.T


def _generate_snippet_indices(
    audio: np.ndarray,
    sr: int,
    snippet_seconds: int,
    count: int,
    gate_db: float = -45.0,
) -> List[int]:
    snippet = max(int(snippet_seconds * sr), 1)
    total = audio.shape[-1]
    stride = max(snippet // 2, int(0.5 * sr))
    rms_threshold = 10 ** (gate_db / 20.0)
    starts: List[int] = []
    cursor = 0
    while cursor + snippet < total and len(starts) < count * 3:
        window = audio[..., cursor : cursor + snippet]
        rms = float(np.sqrt(np.mean(window**2))) + 1e-12
        if rms >= rms_threshold:
            starts.append(cursor)
        cursor += stride
    if not starts:
        return [0]
    step = max(len(starts) // max(count, 1), 1)
    return starts[::step][:count]


class MusicRemovalPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        raw_paths = config.get("paths", {})
        self.paths = {
            key: Path(value) if isinstance(value, str) else Path(value)
            for key, value in raw_paths.items()
        }
        self.input_path = Path(self.paths["input_wav"])
        alias_path = self.paths.get("alias")
        self.base_name = alias_path.as_posix() if alias_path else self.input_path.stem
        self.device = _resolve_device(config.get("separation", {}).get("device"))
        self.metadata: Dict[str, Any] = {
            "input_path": str(self.input_path),
            "base_name": self.base_name,
        }

    @property
    def pre_path(self) -> Path:
        return self.paths["intermediates_dir"] / f"{self.base_name}__pre_loudnorm.wav"

    @property
    def stage_a_path(self) -> Path:
        return (
            self.paths["intermediates_dir"] / f"{self.base_name}__demucs_vocals.wav"
        )

    @property
    def stage_b_path(self) -> Path:
        return (
            self.paths["intermediates_dir"] / f"{self.base_name}__denoiser_stage.wav"
        )

    @property
    def masked_path(self) -> Path:
        return (
            self.paths["intermediates_dir"]
            / f"{self.base_name}__beta_wiener_stage.wav"
        )

    def run(self) -> Dict[str, Any]:
        logging.info("Starting pipeline for %s", self.input_path)
        for path in [
            self.paths["artifacts_dir"],
            self.paths["intermediates_dir"],
            self.paths["separated_dir"],
            self.paths.get("ab_dir", self.paths["artifacts_dir"]),
            self.paths.get("eval_dir", self.paths["artifacts_dir"]),
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

        prep_info = self._preprocess()
        sep_info = self._run_separation()
        enh_info = self._enhance(sep_info)
        final_info = self._postprocess()
        ab_info = self._make_ab_snippets()

        self.metadata.update(prep_info)
        self.metadata.update(sep_info)
        self.metadata.update(enh_info)
        self.metadata.update(final_info)
        self.metadata.update(ab_info)
        self._write_metadata()
        self._append_experiment_row()
        logging.info("Pipeline completed: %s", self.paths["final_wav"])
        return self.metadata

    def _preprocess(self) -> Dict[str, Any]:
        logging.info("Preprocessing and loudness normalizing input.")
        audio, sr = audio_utils.load_audio(self.input_path)
        orig_loudness_audio = audio.T
        meter = pyln.Meter(sr)  # type: ignore[name-defined]
        init_lufs = float(meter.integrated_loudness(orig_loudness_audio))
        cfg = self.config.get("preprocess", {})
        target = cfg.get("loudnorm_target_lufs", -23.0)
        true_peak = cfg.get("loudnorm_true_peak_db", -1.0)
        norm_audio, _ = audio_utils.loudness_normalize(
            audio, sr, target_lufs=target, true_peak_db=true_peak
        )
        audio_utils.save_audio(self.pre_path, norm_audio, sr)
        self.metadata["sample_rate"] = sr
        self.metadata["preprocessed_path"] = str(self.pre_path)
        return {
            "input_lufs": init_lufs,
            "pre_loudnorm_target": target,
            "pre_true_peak_db": true_peak,
        }

    def _run_separation(self) -> Dict[str, Any]:
        cfg = self.config.get("separation", {})
        base_defaults = {k: v for k, v in cfg.items() if k != "models"}
        models_cfg = cfg.get("models") or [base_defaults]
        stem_infos = []
        model_metadata = []
        for entry in models_cfg:
            merged = {**base_defaults, **entry}
            info = self._run_single_model(merged)
            stem_infos.append(info)
            model_metadata.append(
                {
                    "model": info["model"],
                    "vocals_path": info["vocals_path"],
                    "residual_path": info["residual_path"],
                }
            )
        if not stem_infos:
            raise RuntimeError("No separation models configured.")
        blended_vocals, blended_residual, sr = self._blend_stems(stem_infos, cfg)
        audio_utils.save_audio(self.stage_a_path, blended_vocals, sr)
        combined_residual_path = (
            self.paths["intermediates_dir"] / f"{self.base_name}__combined_no_vocals.wav"
        )
        audio_utils.save_audio(combined_residual_path, blended_residual, sr)
        self.metadata["separation_models"] = model_metadata
        return {
            "separation_model": ",".join(m["model"] for m in model_metadata),
            "separation_two_stems": "vocals",
            "separation_vocals": str(self.stage_a_path),
            "separation_residual": str(combined_residual_path),
        }

    def _run_single_model(self, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
        model = model_cfg.get("model_name", "htdemucs")
        two_stems = model_cfg.get("two_stems", "vocals")
        shifts = model_cfg.get("shifts", 2)
        overlap = model_cfg.get("overlap", 0.5)
        segment = model_cfg.get("segment", None)
        jobs = model_cfg.get("jobs", 0)
        device = model_cfg.get("device", self.device)
        reuse = model_cfg.get(
            "reuse_existing",
            self.config.get("separation", {}).get("reuse_existing", True),
        )
        stem_dir = (
            self.paths["separated_dir"]
            / model
            / self.pre_path.stem
        )
        vocals_path = stem_dir / f"{two_stems}.wav"
        residual_path = stem_dir / f"no_{two_stems}.wav"
        if reuse and vocals_path.exists() and residual_path.exists():
            logging.info("Reusing cached Demucs stems (%s) at %s", model, stem_dir)
        else:
            cmd = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                model,
                "-o",
                str(self.paths["separated_dir"]),
                str(self.pre_path),
            ]
            if two_stems:
                cmd.extend(["--two-stems", two_stems])
            if shifts is not None:
                cmd.extend(["--shifts", str(shifts)])
            if overlap is not None:
                cmd.extend(["--overlap", str(overlap)])
            if segment and float(segment) > 0:
                cmd.extend(["--segment", str(segment)])
            cmd.extend(["--device", device if device != "auto" else self.device])
            if jobs:
                cmd.extend(["-j", str(jobs)])
            logging.info("Running Demucs separation (%s): %s", model, " ".join(cmd))
            subprocess.run(cmd, check=True)
        if not vocals_path.exists() or not residual_path.exists():
            raise FileNotFoundError(
                f"Expected stems missing for model {model}: {stem_dir}"
            )
        vocals_audio, sr = audio_utils.load_audio(vocals_path)
        residual_audio, _ = audio_utils.load_audio(residual_path)
        return {
            "model": model,
            "vocals": vocals_audio,
            "residual": residual_audio,
            "sr": sr,
            "vocals_path": str(vocals_path),
            "residual_path": str(residual_path),
        }

    def _blend_stems(
        self, stem_infos: List[Dict[str, Any]], cfg: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        mixture, mix_sr = audio_utils.load_audio(self.pre_path)
        target_sr = stem_infos[0]["sr"]
        if mix_sr != target_sr:
            mixture = librosa.resample(
                mixture, orig_sr=mix_sr, target_sr=target_sr, axis=-1
            )
        channels = stem_infos[0]["vocals"].shape[0]
        if mixture.shape[0] < channels:
            mixture = np.repeat(mixture, channels, axis=0)
        elif mixture.shape[0] > channels:
            mixture = mixture[:channels]
        min_len = min(
            [mixture.shape[-1]] + [info["vocals"].shape[-1] for info in stem_infos]
        )
        mixture = mixture[..., :min_len]
        for info in stem_infos:
            info["vocals"] = info["vocals"][..., :min_len]
            info["residual"] = info["residual"][..., :min_len]
        n_fft = int(cfg.get("fusion_fft", 4096))
        hop = int(cfg.get("fusion_hop", 1024))
        mask = None
        for info in stem_infos:
            channel_masks = []
            for ch in range(channels):
                spec_v = librosa.stft(info["vocals"][ch], n_fft=n_fft, hop_length=hop)
                spec_r = librosa.stft(info["residual"][ch], n_fft=n_fft, hop_length=hop)
                mag_v = np.abs(spec_v)
                mag_r = np.abs(spec_r)
                ch_mask = (mag_v**2) / (mag_v**2 + mag_r**2 + 1e-9)
                channel_masks.append(ch_mask)
            model_mask = np.stack(channel_masks, axis=0)
            mask = model_mask if mask is None else np.maximum(mask, model_mask)
        fused_vocals = []
        fused_residual = []
        for ch in range(channels):
            mix_spec = librosa.stft(mixture[ch], n_fft=n_fft, hop_length=hop)
            ch_mask = np.clip(mask[ch], 0.0, 1.0)
            vocal_spec = ch_mask * mix_spec
            accomp_spec = (1.0 - ch_mask) * mix_spec
            fused_vocals.append(
                librosa.istft(vocal_spec, hop_length=hop, length=min_len)
            )
            fused_residual.append(
                librosa.istft(accomp_spec, hop_length=hop, length=min_len)
            )
        vocals_arr = np.stack(fused_vocals, axis=0).astype(np.float32)
        residual_arr = np.stack(fused_residual, axis=0).astype(np.float32)
        return vocals_arr, residual_arr, target_sr

    def _enhance(self, sep_info: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.get("enhancement", {})
        method = cfg.get("method", "denoiser")
        dry_wet = float(cfg.get("dry_wet", 0.2))
        residual = float(cfg.get("residual_blend", 0.05))
        checkpoint = cfg.get("checkpoint", "dns64")
        logging.info("Enhancing vocals with %s (%s).", method, checkpoint)
        enhancer.enhance(
            input_path=self.stage_a_path,
            output_path=self.stage_b_path,
            method=method,
            checkpoint=checkpoint,
            dry_wet=dry_wet,
            residual_blend=residual,
            device=self.device,
        )
        beta_cfg = cfg.get("beta_wiener", {})
        enhanced_audio, sr = audio_utils.load_audio(self.stage_b_path)
        residual_path = Path(sep_info["separation_residual"])
        if beta_cfg.get("enabled", False) and residual_path.exists():
            logging.info("Applying beta-Wiener soft mask post-filter.")
            residual_audio, _ = audio_utils.load_audio(residual_path)
            masked = _beta_wiener_mask(
                vocals=enhanced_audio,
                accompaniment=residual_audio,
                beta=float(beta_cfg.get("beta", 1.1)),
                floor=float(beta_cfg.get("floor", 0.05)),
                fft_size=int(beta_cfg.get("fft_size", 2048)),
                hop_length=int(beta_cfg.get("hop_length", 512)),
            )
            audio_utils.save_audio(self.masked_path, masked, sr)
            self.current_audio = masked
        else:
            self.current_audio = enhanced_audio
        self.current_sr = sr
        return {
            "enhancer": method,
            "enhancer_checkpoint": checkpoint,
            "enhancer_dry_wet": dry_wet,
        }

    def _postprocess(self) -> Dict[str, Any]:
        cfg = self.config.get("postprocess", {})
        sr = getattr(self, "current_sr", None)
        if sr is None:
            raise RuntimeError("Stage audio not found.")
        audio = self.current_audio
        if cfg.get("highpass_hz", 0) > 0:
            audio = audio_utils.highpass(audio, sr, cfg.get("highpass_hz", 70))
        if cfg.get("high_shelf_db", 0):
            audio = audio_utils.high_shelf(
                audio,
                sr,
                freq=cfg.get("high_shelf_hz", 12000),
                gain_db=cfg.get("high_shelf_db", 1.0),
                q=cfg.get("high_shelf_q", 0.707),
            )
        target_lufs = cfg.get("final_lufs", -16.0)
        true_peak = cfg.get("final_true_peak_db", -1.0)
        audio, loudness = audio_utils.loudness_normalize(
            audio, sr, target_lufs=target_lufs, true_peak_db=true_peak
        )
        audio_utils.save_audio(self.paths["final_wav"], audio, sr)
        self.metadata["final_path"] = str(self.paths["final_wav"])
        self.metadata["final_sr"] = sr
        self.final_audio = audio
        self.final_sr = sr
        return {
            "final_loudness": target_lufs,
            "final_true_peak": true_peak,
        }

    def _make_ab_snippets(self) -> Dict[str, Any]:
        cfg = self.config.get("ab_tests", {})
        seconds = int(cfg.get("snippet_seconds", 12))
        count = int(cfg.get("snippet_count", 3))
        gate_db = float(cfg.get("voiced_gate_db", -45))
        orig_audio, orig_sr = audio_utils.load_audio(self.input_path)
        final_audio = getattr(self, "final_audio", None)
        if final_audio is None:
            raise RuntimeError("Final audio not computed.")
        sr = self.final_sr
        if orig_sr != sr:
            orig_audio_resampled = librosa.resample(
                orig_audio.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=sr,
                axis=-1,
            )
        else:
            orig_audio_resampled = orig_audio
        ab_dir = self.paths.get("ab_dir", self.paths["artifacts_dir"])
        ab_dir.mkdir(parents=True, exist_ok=True)
        indices = _generate_snippet_indices(
            audio=final_audio,
            sr=sr,
            snippet_seconds=seconds,
            count=count,
            gate_db=gate_db,
        )
        written: List[str] = []
        total_len = final_audio.shape[-1]
        for idx, start_sample in enumerate(indices):
            end_sample = min(start_sample + seconds * sr, total_len)
            seg_clean = final_audio[..., start_sample:end_sample]
            seg_orig = orig_audio_resampled[..., start_sample:end_sample]
            clean_path = ab_dir / f"{self.base_name}__segment{idx + 1}_clean.wav"
            orig_path = ab_dir / f"{self.base_name}__segment{idx + 1}_orig.wav"
            audio_utils.save_audio(clean_path, seg_clean, sr)
            audio_utils.save_audio(orig_path, seg_orig, sr)
            written.extend([str(clean_path), str(orig_path)])
        return {"ab_snippets": written}

    def _write_metadata(self) -> None:
        meta_path = self.paths["artifacts_dir"] / f"{self.base_name}__metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(self.metadata, indent=2))

    def _append_experiment_row(self) -> None:
        csv_path = self.paths["artifacts_dir"] / "experiments.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "base_name",
            "model",
            "two_stems",
            "shifts",
            "overlap",
            "enhancer",
            "checkpoint",
            "dry_wet",
            "final_lufs",
        ]
        row = [
            self.base_name,
            self.config.get("separation", {}).get("model_name", "htdemucs"),
            self.config.get("separation", {}).get("two_stems", "vocals"),
            self.config.get("separation", {}).get("shifts", 2),
            self.config.get("separation", {}).get("overlap", 0.5),
            self.config.get("enhancement", {}).get("method", "denoiser"),
            self.config.get("enhancement", {}).get("checkpoint", "dns64"),
            self.config.get("enhancement", {}).get("dry_wet", 0.2),
            self.config.get("postprocess", {}).get("final_lufs", -16.0),
        ]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    config = load_config(args.config)
    pipeline = MusicRemovalPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
