# Pilot music removal report

## Method
- **Stage A – Dual heavy separators.** We now run both `htdemucs_ft` (Transformer finetuned variant, 4 shifts, 6 s segments, 0.55 overlap) and the `mdx_extra_q` UVR checkpoint (2 shifts, 0.25 overlap). Their vocal/accompaniment ratios are converted into STFT masks, max-fused, and reapplied to the loudness-normalized mixture so bleed that survives one model is suppressed by the other. Stems are cached so subsequent iterations only redo the expensive stage if inputs change.
- **Stage B – Speech enhancement** with `facebookresearch/denoiser` (`dns64`, CUDA). We keep a 35% wet mix, blend 5% of the dry signal to preserve ambience, and guard against timing drift/resampling mismatches.
- **Post-filtering**: a beta-Wiener mask (`beta=1.5`, `floor=0.02`, `n_fft=4096`) derived from the fused accompaniment, plus a 70 Hz HPF, 12 kHz shelf, and LUFS normalization to −16 LUFS / −1 dBTP. Loudness is still staged at −23 LUFS pre-separation via `pyloudnorm`.
- **Pre/post loudness** handled in Python (`pyloudnorm`) to guarantee −23 LUFS input staging and reproducible −16 LUFS delivery.

## Results (artifacts/eval/metrics.json)
- **Music-to-speech energy** (masked proxy): **−32.7 dB** (target ≤ −20 dB). The dual-model fusion buys ~15 dB more suppression versus Demucs-only.
- **WER (orig→clean)**: 5.9% (slightly worse than the best Demucs-only run; stakeholders asked us to pause WER tuning until Görkem/Harun wire their metrics, so keeping this noted for later).
- **Speech preservation**: SI-SDR vs fused Stage A `9.36 dB`, STOI `0.996`. Subjectively, speech timbre stays natural with no gating.
- **Music presence classifier (PANNs) mean probability**: 0.015 (clean) vs 0.266 (original) across voiced windows.
- **Loudness**: input −26.0 LUFS, output −16.3 LUFS (−1 dBTP headroom).

AB snippets for rapid review live in `artifacts/ab/segment*_orig/clean.wav`. The full cleaned pilot file is `artifacts/text_batched_generated__speech_only.wav`.

## Reproducibility
1. `python3.10 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `bash run.sh`

`run.sh` executes the full pipeline (`pipeline.py`) and evaluation suite (`evaluate.py`). Configuration, paths, and hyper-parameters are stored in `config/best_pipeline.yaml`.

## Next steps
- Subjective check the AB pairs; if approved, run `python scripts/batch_process.py --config config/best_pipeline.yaml --in-dir data/batch --out-dir artifacts/cleaned` to process the remaining files and zip them.
- Keep an eye on GPU VRAM if you raise `--shifts` or change the Demucs backbone; HTDemucs needs ≤7.8 s segments per repo guidance (see `report/demucs_notes.md`).
