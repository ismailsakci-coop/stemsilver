### 🎧 Speech-Only Rendering Pipeline

This repo contains an end-to-end pipeline for stripping background music from large batches of TTS outputs using a fusion of heavy separator models (HTDemucs-FT + MDX-Extra-Q), DNS64 denoising, beta-Wiener post-masking, and loudness management. Everything you need to reproduce the pilot + batch run lives here.

---

#### 🗂️ Top-Level Layout

```text
.
├── artifacts/              # Staging for stems, AB snippets, metrics, cleaned WAVs, zipped deliveries
├── config/                 # YAML configs (best_pipeline.yaml drives everything)
├── data/                   # Input WAVs (pilot + batch after unzip)
├── notebooks/              # Jupyter diagnostics (e.g., pilot_analysis.ipynb)
├── report/                 # Summary + Demucs research notes
├── scripts/                # Helpers: enhancement, audio I/O, batch driver
├── pipeline.py             # Core orchestration (preprocess → fusion → enhance → post)
├── evaluate.py             # Objective metrics (WER, STOI, SI-SDR, music residuals)
├── run.sh                  # One-command pilot run (pipeline + evaluate)
└── requirements.txt        # Fully pinned Python deps
```

> **Note:** `artifacts/outputs_cleaned.zip` (≈215 MB) is not pushable to GitHub without Git LFS. Keep it locally or upload via release assets.

---

#### 🧠 Core Idea

We treat each separator as providing a soft ratio mask. For separator \(i\) with vocal magnitude \(V_i\) and accompaniment magnitude \(A_i\), we build a fused mask on the mixture STFT \(X\):

\[
M_\text{fused}(f, t) = \max_i \left( \frac{|V_i(f,t)|^2}{|V_i(f,t)|^2 + |A_i(f,t)|^2 + \varepsilon} \right), \qquad \hat{V} = M_\text{fused} \cdot X
\]

This “max fusion” preserves speech details that one model captures better than the other while aggressively nulling accompaniment energy.

---

#### ⚙️ Usage Cheatsheet

| Task | Command |
|------|---------|
| Create venv + install deps | `python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| Pilot run (single file) | `bash run.sh` |
| Batch all WAVs under `data/batch/outputs/` | `python -m scripts.batch_process --config config/best_pipeline.yaml --in-dir data/batch/outputs --pattern '**/*.wav' --out-dir artifacts/cleaned` |
| Re-run evaluation only | `python evaluate.py --config config/best_pipeline.yaml --out-dir artifacts/eval` |

**Pipeline knobs:** `config/best_pipeline.yaml` exposes per-stage hyper-parameters—add/remove separator entries, change overlaps, tweak denoiser wet mix, etc. Each batch item uses an alias such as `seed_42_run1__text_batched_generated` so its intermediates and metadata stay isolated.

---

#### 📊 Metrics & Reporting

- Objective scores (LUFS, masked music-to-speech ratio, SI-SDR, STOI, WER, PANNs music presence) land in `artifacts/eval/metrics.json` plus `metrics.csv`.
- AB snippets (`artifacts/ab/…`) allow quick subject listening.
- `report/summary.md` captures the latest run configuration and highlights; `report/demucs_notes.md` is a short research digest from the Demucs repository review.

---

#### 🔬 Notebooks

`notebooks/pilot_analysis.ipynb` plots waveform and spectrogram comparisons for the pilot. Extend it to inspect any of the cleaned batch files—just change the paths at the top.

---

#### ✅ Tips

- Keep `outputs.zip` archived (in `data/`), but unzip fresh under `data/batch/outputs/` before running the batch script.
- For future pushes, consider turning on Git LFS for the cleaned ZIP if you need to host it on GitHub (`git lfs install && git lfs track artifacts/outputs_cleaned.zip`).
- GPU VRAM usage: HTDemucs-FT with 4 shifts and 6 s segments fits comfortably on an RTX 4050; MDX-Extra-Q leverages DiffQ (already installed) so expect a short model download during the first run.

---

#### 📦 What This Pipeline Delivers

| Stage | Components | Purpose / Outcome |
|-------|------------|-------------------|
| **Stage A** | HTDemucs-FT (4 shifts, 6 s segments) + MDX-Extra-Q | Heavy separators run in parallel, their ratio masks are max-fused so speech cues from either model survive while accompaniment energy is nulled. |
| **Stage B** | DNS64 denoiser (35 % wet mix, 5 % dry blend) | Removes musical bleed and broadband haze while preserving room tone. |
| **Post** | Beta-Wiener (β = 1.5), EQ, LUFS normalization | Final noise/mask polish, 70 Hz HPF, slight 12 kHz shelf, render at −16 LUFS / −1 dBTP for spoken-word delivery. |
| **Evaluation** | Whisper large-v3 (WER), STOI, SI-SDR, PANNs | Objective verification per file; AB snippets prepared for listening tests. |
| **Batch Ops** | `scripts.batch_process` + aliasing | Processes every WAV under `data/batch/outputs`, mirroring folder structure inside `artifacts/cleaned/` and caching stems for fast re-runs. |

**Headline metrics (pilot file `text_batched_generated.wav`):**

- Music-to-Speech Energy (masked proxy): **−32.7 dB** (target ≤ −20 dB)
- STOI vs fused vocals: **0.996**
- SI-SDR vs fused vocals: **9.36 dB**
- WER (orig → clean): **5.9 %** (to be revisited when Görkem/Harun provide the target ASR)

**Overall workflow summary**

1. Normalize each mixture to −23 LUFS, keeping true peak ≤ −1 dBTP.
2. Run the dual separators → fuse masks → derive 1 mixed vocal stem + 1 accompaniment.
3. Apply DNS64 + beta-Wiener to suppress residual music.
4. Re-normalize to −16 LUFS, export cleaned WAVs, AB snippets, metadata, and evaluation charts.
5. Zip the batch (`artifacts/outputs_cleaned.zip`) for downstream delivery.

Happy separating! 🎶➝🗣️
