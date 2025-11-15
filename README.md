### 🎧 Speech-Only Rendering Pipeline

<p align="center">
  <img src="logo/stemsilver_logo.svg" width="190" alt="Stemsilver logo">
</p>

This repo contains an end-to-end pipeline for removing backing music from TTS outputs using HTDemucs-FT, MDX-Extra-Q, DNS64 denoiser, beta-Wiener post-masking, and LUFS normalization.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=ffffff">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=ffffff">
  <img alt="Bash" src="https://img.shields.io/badge/Bash-scripts-4EAA25?style=for-the-badge&logo=gnubash&logoColor=ffffff">
  <img alt="ffmpeg" src="https://img.shields.io/badge/ffmpeg-post-0A7B1D?style=for-the-badge&logo=ffmpeg&logoColor=ffffff">
  <img alt="Mermaid" src="https://img.shields.io/badge/Docs-Mermaid-00B4B6?style=for-the-badge&logo=markdown&logoColor=ffffff">
</p>

| Layer | Stack | Highlights |
|-------|-------|------------|
| Separation | Python 3.10 · PyTorch 2.5 · Torchaudio · Demucs/MDX | GPU-accelerated hybrid masking with HTDemucs-FT + MDX-Extra-Q |
| Enhancement | DNS64 · Beta-Wiener · ffmpeg | Speech denoising + LUFS shaping |
| Tooling | Bash · Git LFS · Mermaid docs | Batch scripts, Colab notebooks, narrative docs |

#### Contents
- [Layout](#-layout)
- [Colab heavy run](#colab-heavy-run)
- [Core Idea](#-core-idea)
- [Usage Cheatsheet](#-usage-cheatsheet)
- [Quick Listen](#-quick-listen)
- [Pipeline Highlights & Metrics](#-pipeline)
- [Improvement roadmap](#-todo-roadmap)

### Colab Heavy Run

| Parameter | Value | Why |
|-----------|-------|-----|
| `--shifts` | 100 | Maximizes HTDemucs equivariant averaging to suppress music beds |
| `--segment` | 7 sec | Keeps within Transformer context window while fitting in Colab GPU RAM |
| `--overlap` | 0.95 | Dense overlap to avoid stitching artefacts |
| Runtime | ≈45 min on Colab A100 | Includes upload/download overhead |

Outputs are stored under `google_colab_runned/` (`hevay_work.ipynb` notebook + `separated_audio_htdemucs_ft_optimized/`).

**Quick listen (Colab run):**

| Stem | Player |
|------|--------|
| text_batched_generated.wav | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/google_colab_runned/separated_audio_htdemucs_ft_optimized/htdemucs_ft/text_batched_generated/text_batched_generated.wav) |
| music_combined_optimized.wav | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/google_colab_runned/separated_audio_htdemucs_ft_optimized/htdemucs_ft/text_batched_generated/music_combined_optimized.wav) |
| vocals.wav | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/google_colab_runned/separated_audio_htdemucs_ft_optimized/htdemucs_ft/text_batched_generated/vocals.wav) |

---

#### 🗂️ Layout
```text
.
├── artifacts/
├── config/
├── data/
├── notebooks/
├── report/
├── scripts/
├── pipeline.py
├── evaluate.py
├── run.sh
└── requirements.txt
```

---

#### 🧠 Core Idea
We treat each separator as providing a soft ratio mask. For separator $i$ with vocal magnitude $V_i$ and accompaniment magnitude $A_i$, we build a fused mask on the mixture STFT $X$:
$$M_\text{fused}(f,t) = \max_i \left( \frac{|V_i(f,t)|^2}{|V_i(f,t)|^2 + |A_i(f,t)|^2 + \varepsilon} \right), \qquad \hat{V}=M_\text{fused}\cdot X$$

---

#### ⚙️ Usage Cheatsheet
| Task | Command |
|------|---------|
| Create venv + install deps | `python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| Pilot run | `bash run.sh` |
| Batch WAVs | `python -m scripts.batch_process --config config/best_pipeline.yaml --in-dir data/batch/outputs --pattern '**/*.wav' --out-dir artifacts/cleaned` |
| Evaluate a pair | `python evaluate.py --config config/best_pipeline.yaml --out-dir artifacts/eval` |

---

#### 🔊 Quick Listen
| Original mix | Clean render |
|--------------|--------------|
| [▶️ Listen](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/data/text_batched_generated.wav) | [▶️ Listen](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/artifacts/text_batched_generated__speech_only.wav) |

**Batch comparisons:**

| File | Raw | Cleaned |
|------|-----|---------|
| 1.5b_text_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/1.5b_text_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/1.5b_text_generated__speech_only.wav) |
| 2p_goat_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/2p_goat_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/2p_goat_generated__speech_only.wav) |
| 7b_text_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/7b_text_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/7b_text_generated__speech_only.wav) |
| text_generated_compiled | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/text_generated_compiled.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/text_generated_compiled__speech_only.wav) |
| sequential/text_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/sequential/text_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/sequential/text_generated__speech_only.wav) |
| batched/text_batched_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/batched/text_batched_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/batched/text_batched_generated__speech_only.wav) |
| no_seed/text_batched_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/no_seed/text_batched_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/no_seed/text_batched_generated__speech_only.wav) |
| seed_42_run1/text_batched_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/seed_42_run1/text_batched_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/seed_42_run1/text_batched_generated__speech_only.wav) |
| seed_42_run2/text_batched_generated | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/raw/outputs/seed_42_run2/text_batched_generated.wav) | [▶️](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/ismailsakci-coop/stemsilver/main/proceed/seed_42_run2/text_batched_generated__speech_only.wav) |

---

#### 📦 Pipeline
| Stage | Components | Outcome |
|-------|------------|---------|
| Stage A | HTDemucs-FT (4 shifts, 6 s), MDX-Extra-Q | Dual models, max-fused ratio masks applied to mix |
| Stage B | DNS64 denoiser (35 % wet, 5 % dry blend) | Removes residual music/hiss, keeps room tone |
| Post | Beta-Wiener (β=1.5), 70 Hz HPF, 12 kHz shelf, LUFS | Clean polish at −16 LUFS / −1 dBTP |
| Evaluation | Whisper large-v3, STOI, SI-SDR, PANNs | Objective proof: speech intact, music suppressed |

```mermaid
flowchart LR
  M((Mixture WAV)) --> |Loudnorm -23 LUFS| P1([Preprocess]) --> P2([HTDemucs-FT])
  P2 -->|ratio mask| F((Fusion))
  MDX([MDX-Extra-Q]) -->|ratio mask| F
  F --> D([DNS64 + Beta-Wiener]) --> PP([EQ + LUFS -16]) --> OUT((Speech-only WAV))
  OUT --> EVAL([Whisper / STOI / SI-SDR])

  classDef default fill:#edf2f4,stroke:#2b2d42,stroke-width:2px,color:#2b2d42;
  classDef io fill:#8d99ae,stroke:#2b2d42,color:#ffffff,stroke-width:2px;
  classDef core fill:#a2d2ff,stroke:#023047,color:#023047;
  classDef post fill:#ffc8dd,stroke:#b8306b,color:#3a0d2c;
  classDef eval fill:#cdb4db,stroke:#6d597a,color:#2b2d42;

  class M,OUT io;
  class P2,MDX,F core;
  class D,PP post;
  class EVAL eval;
```

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
| SI-SDR | — | 9.36 dB |
| WER | Reference | 5.9 % |

<table>
  <thead>
    <tr>
      <th style="text-align:left">File</th>
      <th style="text-align:center">Music ↦ Speech (dB)</th>
      <th style="text-align:center">STOI</th>
      <th style="text-align:center">SI-SDR (dB)</th>
      <th style="text-align:center">WER</th>
      <th style="text-align:left">Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>text_batched_generated.wav</code></td>
      <td style="text-align:center">−32.7</td>
      <td style="text-align:center">0.996</td>
      <td style="text-align:center">9.36</td>
      <td style="text-align:center">5.9 %</td>
      <td>Pilot reference – demonstrates current best pipeline.</td>
    </tr>
    <tr>
      <td><em>Batch items</em></td>
      <td style="text-align:center">TBD</td>
      <td style="text-align:center">TBD</td>
      <td style="text-align:center">TBD</td>
      <td style="text-align:center">TBD</td>
      <td>Populate by running <code>scripts/evaluate.py</code> on each cleaned file.</td>
    </tr>
  </tbody>
</table>

---

#### ✅ TODO Roadmap

```mermaid
flowchart LR
  subgraph EXPERIMENTS[ ]
    direction TB
    A1(["HTDemucs variants"]):::experiment
    A2(["MDX / UVR fusion"]):::experiment
    A3(["Post-mask research"]):::experiment
    A1 --> A2 --> A3
  end

  subgraph AUTOMATION[ ]
    direction TB
    B1(["Batch metrics job"]):::auto
    B2(["WER guardrail CI"]):::auto
    B3(["Release scorecard"]):::auto
    B1 --> B2 --> B3
  end

  subgraph DELIVERY[ ]
    direction TB
    C1(["Stakeholder microsite"]):::delivery
    C2(["Pages audio gallery"]):::delivery
    C3(["One-click ZIP publish"]):::delivery
    C1 --> C2 --> C3
  end

  EXPERIMENTS --> AUTOMATION --> DELIVERY

  classDef experiment fill:#8ecae6,stroke:#023047,color:#072b3d,stroke-width:2px;
  classDef auto fill:#ffb703,stroke:#bb3e03,color:#023047,stroke-width:2px;
  classDef delivery fill:#219ebc,stroke:#023047,color:#e0fbfc,stroke-width:2px;
```

| Status | Task |
|--------|------|
| ⬜ | Evaluate MDX-UVR HQ3 + SoftMasking vs current fusion. |
| ⬜ | Build a GitHub Action that recomputes LUFS/STOI/WER for every PR. |
| ⬜ | Add `artifacts/eval/batch_metrics.csv` by running `evaluate.py` on each cleaned file. |
| ⬜ | Publish a GitHub Pages microsite hosting the gabalpha players for stakeholder review.
