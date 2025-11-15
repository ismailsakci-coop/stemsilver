### 🎧 Speech-Only Rendering Pipeline

This repo contains an end-to-end pipeline for removing backing music from TTS outputs using HTDemucs-FT, MDX-Extra-Q, DNS64 denoiser, beta-Wiener post-masking, and LUFS normalization.

---

#### 🗂️ Layout
```text
.
├── artifacts/              # Staging for stems, AB snippets, metrics, cleaned WAVs, zipped deliveries
├── config/                 # YAML configs (best_pipeline.yaml drives everything)
├── data/                   # Input WAVs (pilot + batch after unzip)
├── notebooks/              # Diagnostics notebooks
├── report/                 # Summary + Demucs notes
├── scripts/                # Helpers: enhancement, audio I/O, batch driver
├── pipeline.py             # Core orchestration (preprocess → fusion → enhance → post)
├── evaluate.py             # Objective metrics (WER, STOI, SI-SDR, music residuals)
├── run.sh                  # One-command pilot run (pipeline + evaluate)
└── requirements.txt        # Fully pinned Python deps
```

---

#### 🧠 Core Idea
We treat each separator as providing a soft ratio mask. For separator $i$ with vocal magnitude $V_i$ and accompaniment magnitude $A_i$, we build a fused mask on the mixture STFT $X$:
$$
M_\text{fused}(f,t) = \max_i \left( \frac{|V_i(f,t)|^2}{|V_i(f,t)|^2 + |A_i(f,t)|^2 + \varepsilon} \right), \qquad \hat{V}=M_\text{fused}\cdot X
$$
This “max fusion” keeps whichever model best captures a speech component, while suppressing accompaniment.

---

#### ⚙️ Usage Cheatsheet
| Task | Command |
|------|---------|
| Create venv + install deps | `python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| Pilot run | `bash run.sh` |
| Batch WAVs (mirror folder structure) | `python -m scripts.batch_process --config config/best_pipeline.yaml --in-dir data/batch/outputs --pattern '**/*.wav' --out-dir artifacts/cleaned` |
| Evaluate a pair | `python evaluate.py --config config/best_pipeline.yaml --out-dir artifacts/eval` |

---

#### 🔊 Quick Listen
| Original mix | Clean render |
|--------------|--------------|
| [▶️ Listen](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/sep/main/data/text_batched_generated.wav) | [▶️ Listen](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/sep/main/artifacts/text_batched_generated__speech_only.wav) |

GitHub README'leri doğrudan WAV/MP4 oynatamadığı için gabalpha'nın hafif player'ını kullanıyoruz; bağlantılar anında ses çalıyor.

---

#### 📦 Pipeline Highlights
| Stage | Components | Outcome |
|-------|------------|---------|
| Stage A | HTDemucs-FT (4 shifts, 6 s), MDX-Extra-Q | Dual models, max-fused ratio masks applied to mix |
| Stage B | DNS64 denoiser (35 % wet, 5 % dry blend) | Removes residual music/hiss, keeps room tone |
| Post | Beta-Wiener (β=1.5), 70 Hz HPF, 12 kHz shelf, LUFS | Clean polish at −16 LUFS / −1 dBTP |
| Evaluation | Whisper large-v3, STOI, SI-SDR, PANNs | Objective proof: speech intact, music suppressed |

**Pilot metrics (`text_batched_generated.wav`):**
- Masked music-to-speech: −32.7 dB
- STOI vs fused vocals: 0.996
- SI-SDR vs fused vocals: 9.36 dB
- WER (orig → clean): 5.9 %

| Metric | Original | Cleaned |
|--------|----------|---------|
| LUFS | −26.0 | −16.3 |
| Music ↦ Speech energy | 0 dB | −32.7 dB |
| STOI | — | 0.996 |
| SI-SDR | — | 9.36 dB |
| WER | Reference | 5.9 % |

---

#### 📊 Reporting & Notebooks
- `artifacts/eval/metrics.json` – full metrics dump
- `artifacts/ab/*` – AB snippets
- `report/summary.md` – methods & results
- `notebooks/pilot_analysis.ipynb` – waveform + spectrogram comparisons

---

#### ✅ Tips
- Always unzip new batches into `data/batch/outputs/`, keeping the folder hierarchy intact; `scripts.batch_process` mirrors it to `artifacts/cleaned/`.
- Use Git LFS (already enabled) to push large artifacts such as `artifacts/outputs_cleaned.zip`.
- GPU headroom: HTDemucs-FT with 4 shifts fits comfortably on RTX 4050; MDX-Extra-Q (DiffQ dependency) streams under 6 GB.

Happy separating 🎶➝🗣️
