from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
import torchaudio
import torchaudio.functional as AF

try:
    from denoiser import pretrained as denoiser_pretrained
except ImportError as exc:  # pragma: no cover
    denoiser_pretrained = None
    raise exc


EnhancerMethod = Literal["denoiser"]


def _resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_denoiser(checkpoint: str, device: str) -> torch.nn.Module:
    if denoiser_pretrained is None:
        raise RuntimeError("facebookresearch/denoiser is not installed")
    checkpoint = checkpoint.lower()
    if checkpoint == "dns64":
        model = denoiser_pretrained.dns64()
    elif checkpoint == "dns48":
        model = denoiser_pretrained.dns48()
    elif checkpoint == "master64":
        model = denoiser_pretrained.master64()
    else:
        raise ValueError(f"Unsupported denoiser checkpoint: {checkpoint}")
    model.eval()
    model.to(device)
    return model


def denoiser_enhance(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint: str = "dns64",
    dry_wet: float = 0.15,
    residual_blend: float = 0.05,
    device: str | None = None,
) -> str:
    """Apply facebookresearch/denoiser to the waveform."""
    device = _resolve_device(device)
    waveform, sr = torchaudio.load(str(input_path))
    model = _load_denoiser(checkpoint, device)
    target_sr = getattr(model, "sample_rate", 16000)
    orig_waveform = waveform.clone()
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    with torch.no_grad():
        enhanced = model(waveform.to(device))
    enhanced = enhanced.squeeze(1).cpu() if enhanced.dim() == 3 else enhanced.cpu()
    if sr != target_sr:
        enhanced = AF.resample(enhanced, target_sr, sr)
    if enhanced.shape[-1] != orig_waveform.shape[-1]:
        min_len = min(enhanced.shape[-1], orig_waveform.shape[-1])
        enhanced = enhanced[..., :min_len]
        orig_waveform = orig_waveform[..., :min_len]
    dry = float(max(0.0, min(1.0, dry_wet)))
    enhanced = dry * enhanced + (1.0 - dry) * orig_waveform
    if residual_blend > 0:
        enhanced = torch.tanh(enhanced + residual_blend * orig_waveform)
    enhanced = torch.clamp(enhanced, -1.0, 1.0)
    if enhanced.dim() == 1:
        enhanced = enhanced.unsqueeze(0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), enhanced, sr)
    return str(output_path)


def enhance(
    input_path: str | Path,
    output_path: str | Path,
    method: EnhancerMethod = "denoiser",
    checkpoint: str = "dns64",
    dry_wet: float = 0.2,
    residual_blend: float = 0.05,
    device: str | None = None,
) -> str:
    if method != "denoiser":
        raise ValueError(f"Unsupported method: {method}")
    return denoiser_enhance(
        input_path=input_path,
        output_path=output_path,
        checkpoint=checkpoint,
        dry_wet=dry_wet,
        residual_blend=residual_blend,
        device=device,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance a vocals stem")
    parser.add_argument("--in", dest="input_path", required=True, help="Input WAV path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output WAV path")
    parser.add_argument("--method", default="denoiser", choices=["denoiser"], help="Enhancer backend")
    parser.add_argument("--checkpoint", default="dns64", help="Denoiser checkpoint")
    parser.add_argument("--dry-wet", type=float, default=0.2, help="Wet mix ratio")
    parser.add_argument(
        "--residual-blend",
        type=float,
        default=0.05,
        help="Blend amount of original waveform to preserve room tone",
    )
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    enhance(
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        checkpoint=args.checkpoint,
        dry_wet=args.dry_wet,
        residual_blend=args.residual_blend,
        device=args.device,
    )


if __name__ == "__main__":
    main()
