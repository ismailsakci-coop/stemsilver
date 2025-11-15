from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import yaml
from faster_whisper import WhisperModel
from jiwer import wer
from panns_inference import AudioTagging, labels as panns_labels
from pystoi import stoi


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    return data.astype(np.float32).T, sr


def _energy_db(audio: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(np.square(audio))) + 1e-12)


def _calc_music_ratio(clean: np.ndarray, music: np.ndarray) -> float:
    speech_db = _energy_db(clean)
    music_db = _energy_db(music)
    return music_db - speech_db


def _estimate_masked_music_ratio(
    clean: np.ndarray,
    stage_a: np.ndarray,
    residual: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop_length: int = 512,
) -> float:
    clean_ch = clean[0]
    stage_ch = stage_a[0]
    residual_ch = residual[0]
    speech_spec = librosa.stft(stage_ch, n_fft=n_fft, hop_length=hop_length)
    residual_spec = librosa.stft(residual_ch, n_fft=n_fft, hop_length=hop_length)
    spec_clean = librosa.stft(clean_ch, n_fft=n_fft, hop_length=hop_length)
    mag_speech = np.abs(speech_spec)
    mag_residual = np.abs(residual_spec)
    mask = mag_residual / (mag_residual + mag_speech + 1e-9)
    music_est_spec = mask * spec_clean
    music_est = librosa.istft(
        music_est_spec, hop_length=hop_length, length=len(clean_ch)
    )
    return _calc_music_ratio(clean_ch[np.newaxis, :], music_est[np.newaxis, :])


def _si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    if reference.ndim > 1:
        reference = reference[0]
    if estimation.ndim > 1:
        estimation = estimation[0]
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    dot = np.dot(estimation, reference)
    target = dot * reference / (np.linalg.norm(reference) ** 2 + 1e-12)
    noise = estimation - target
    return 10 * np.log10((np.sum(target**2) + 1e-12) / (np.sum(noise**2) + 1e-12))


def _transcribe(model: WhisperModel, path: Path) -> Tuple[str, Dict[str, Any]]:
    segments, info = model.transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
    }


def _music_presence_prob(model: AudioTagging, audio: np.ndarray, sr: int, chunk_s: int, focus: List[str]) -> Dict[str, float]:
    resampled = audio
    if sr != 32000:
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=32000, axis=-1)  # type: ignore[name-defined]
        sr = 32000
    chunk = max(int(chunk_s * sr), sr)
    probs = []
    for start in range(0, resampled.shape[-1], chunk):
        segment = resampled[..., start : start + chunk]
        if segment.shape[-1] < chunk:
            pad = np.zeros((segment.shape[0], chunk - segment.shape[-1]), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=-1)
        audio_tensor = torch.from_numpy(segment).mean(dim=0, keepdim=True).numpy()
        clipwise, _ = model.inference(audio_tensor)
        focus_indices = [i for i, name in enumerate(panns_labels) if name in focus]
        prob = float(np.mean(clipwise[0][focus_indices])) if focus_indices else float(np.mean(clipwise))
        probs.append(prob)
    if not probs:
        probs = [0.0]
    return {
        "mean": float(np.mean(probs)),
        "std": float(np.std(probs)),
        "n_chunks": len(probs),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate music removal output.")
    parser.add_argument("--config", default="config/best_pipeline.yaml", help="Pipeline config for defaults.")
    parser.add_argument("--orig", help="Original mixture wav.")
    parser.add_argument("--clean", help="Cleaned wav.")
    parser.add_argument("--demucs-vocals", help="Path to Demucs vocals stem.")
    parser.add_argument("--demucs-music", help="Path to Demucs no_vocals stem.")
    parser.add_argument("--out-dir", default="artifacts/eval", help="Directory to store metrics.")
    parser.add_argument("--skip-asr", action="store_true", help="Skip Whisper ASR evaluation.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg.get("paths", {})
    evaluation_cfg = cfg.get("evaluation", {})
    base_name = Path(paths["input_wav"]).stem
    meta_path = Path(paths.get("artifacts_dir", "artifacts")) / f"{base_name}__metadata.json"
    metadata: Dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    orig_path = Path(args.orig or metadata.get("input_path") or paths["input_wav"])
    clean_path = Path(args.clean or metadata.get("final_path") or paths["final_wav"])
    default_vocals = metadata.get(
        "demucs_vocals_path",
        str(Path(paths.get("intermediates_dir", "artifacts/intermediates")) / f"{base_name}__demucs_vocals.wav"),
    )
    default_music = metadata.get("demucs_residual_path")
    if not default_music:
        default_music = str(
            Path(paths.get("separated_dir", "artifacts/separated"))
            / cfg.get("separation", {}).get("model_name", "htdemucs")
            / f"{base_name}__pre_loudnorm"
            / "no_vocals.wav"
        )
    demucs_vocals = Path(args.demucs_vocals or default_vocals)
    demucs_music = Path(args.demucs_music or default_music)
    out_dir = Path(args.out_dir or paths.get("eval_dir", "artifacts/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for candidate in [orig_path, clean_path, demucs_vocals, demucs_music]:
        if not Path(candidate).exists():
            raise FileNotFoundError(f"Missing required audio: {candidate}")

    clean_audio, clean_sr = _load_audio(clean_path)
    orig_audio, orig_sr = _load_audio(orig_path)
    stage_a_audio, _ = _load_audio(demucs_vocals)
    music_audio, _ = _load_audio(demucs_music)

    if clean_sr != orig_sr:
        orig_audio = librosa.resample(orig_audio, orig_sr=orig_sr, target_sr=clean_sr, axis=-1)  # type: ignore[name-defined]
        orig_sr = clean_sr
    if stage_a_audio.shape[-1] != clean_audio.shape[-1]:
        stage_a_audio = librosa.util.fix_length(stage_a_audio, clean_audio.shape[-1], axis=-1)  # type: ignore[name-defined]
    if music_audio.shape[-1] != clean_audio.shape[-1]:
        music_audio = librosa.util.fix_length(music_audio, clean_audio.shape[-1], axis=-1)  # type: ignore[name-defined]

    loudness_meter = pyln.Meter(clean_sr)
    orig_lufs = float(loudness_meter.integrated_loudness(orig_audio.T))
    clean_lufs = float(loudness_meter.integrated_loudness(clean_audio.T))

    music_ratio = _estimate_masked_music_ratio(clean_audio, stage_a_audio, music_audio, clean_sr)
    speech_preservation_si_sdr = _si_sdr(stage_a_audio, clean_audio)
    speech_stoi = float(stoi(stage_a_audio[0], clean_audio[0], clean_sr, extended=False))

    metrics: Dict[str, Any] = {
        "orig_lufs": orig_lufs,
        "clean_lufs": clean_lufs,
        "music_to_speech_db": music_ratio,
        "speech_preservation_si_sdr": speech_preservation_si_sdr,
        "speech_stoi_vs_stage_a": speech_stoi,
    }

    if not args.skip_asr:
        model_name = evaluation_cfg.get("asr_model", "large-v3")
        compute_type = evaluation_cfg.get("asr_compute_type", "float16")
        device = evaluation_cfg.get("asr_device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                compute_type = "int8"
        whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        orig_text, orig_info = _transcribe(whisper_model, orig_path)
        clean_text, clean_info = _transcribe(whisper_model, clean_path)
        wer_value = wer(orig_text.lower(), clean_text.lower())
        metrics.update(
            {
                "wer_orig_vs_clean": float(wer_value),
                "orig_transcript": orig_text,
                "clean_transcript": clean_text,
                "orig_asr_meta": orig_info,
                "clean_asr_meta": clean_info,
            }
        )

    focus_labels = evaluation_cfg.get("music_focus_labels", ["Music"])
    chunk_s = evaluation_cfg.get("music_chunk_s", 5)
    tagger = AudioTagging(device="cuda" if torch.cuda.is_available() else "cpu")
    clean_music_prob = _music_presence_prob(tagger, clean_audio, clean_sr, chunk_s, focus_labels)
    orig_music_prob = _music_presence_prob(tagger, orig_audio, orig_sr, chunk_s, focus_labels)
    metrics["music_presence_clean"] = clean_music_prob
    metrics["music_presence_orig"] = orig_music_prob

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for key, value in metrics.items():
            if isinstance(value, (dict, list)):
                writer.writerow([key, json.dumps(value)])
            else:
                writer.writerow([key, value])


if __name__ == "__main__":
    main()
