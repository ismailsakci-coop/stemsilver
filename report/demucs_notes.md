# HTDemucs usage notes

Insights gathered from reviewing https://github.com/facebookresearch/demucs:

- **Segment limits:** Hybrid Transformer Demucs (HTDemucs) constrains the `--segment` argument to ≤7.8 s because the Transformer operates on fixed-length latent windows (see `demucs/hybrid.py`). To process longer tracks we must use overlap-add tiling driven by the CLI `--segment` option; empirically `7` seconds with `--overlap 0.5` keeps GPU usage low.
- **Equivariant shifts:** The `--shifts` flag performs random circular shifts of the input and averages the predictions, mirroring the equivariant stabilization described in the Demucs paper. Even small values (e.g., 2) reduce musical bleed without doubling GPU memory because inference is sequential.
- **Two-stem mode:** `--two-stems vocals` runs Demucs in a constrained mode that returns `{vocals, no_vocals}` stems. This is essentially a soft binary mask derived from the four-stem model weights, saving disk while keeping the accompaniment estimate necessary for later Wiener filtering.
- **Output path layout:** The CLI saves results under `<out>/<model>/<track_name>/<stem>.wav`, where `track_name` is the basename of the processed file. Pre-/post-processing scripts should therefore use deterministic staging names (e.g., `*_pre_loudnorm`) to be able to rehydrate the stems later.
- **Device auto-selection:** The CLI internally defaults to CUDA when available, but explicitly passing `--device cuda` avoids falling back to CPU when `CUDA_VISIBLE_DEVICES` is unset. The repo also documents how to pass `DEMUX_EVALUATE=1` for deterministic benchmarking, which we mirror by fixing seeds in our pipeline.
