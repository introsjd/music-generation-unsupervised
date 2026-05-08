# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425 — Neural Networks, Spring 2026
**Section:** 5
**Institution:** [BRAC University]

---

## Project Overview

This project implements an unsupervised deep learning framework for symbolic music generation using the **MAESTRO v3.0.0 classical piano MIDI dataset**. Rather than relying on labeled data, our models learn the structure, patterns, and style of music entirely on their own — without any genre tags or annotations.

We design, train, and evaluate three progressively complex generative architectures:

| Task | Model | Key Concept | Primary Metric |
|------|-------|-------------|----------------|
| Task 1 | LSTM Autoencoder | Compress → Reconstruct | Reconstruction Loss (BCE) |
| Task 2 | Variational Autoencoder (VAE) | Probabilistic Latent Space | ELBO + KL Divergence |
| Task 3 | Transformer Decoder | Autoregressive Generation | Perplexity |

All three models are compared against two baselines — a **Random Note Generator** and a **Markov Chain model** — using quantitative metrics and generated MIDI audio output.

---

##  Team Members & Contributions

### Sajid Sarower
**Student ID:** 22201983 | **Section:** 5

- Dataset collection, organization, and Google Drive setup
- Exploratory Data Analysis (EDA) — piano roll visualization, note distribution analysis
- Complete preprocessing pipeline — MIDI to piano roll conversion, segmentation, dataset building
- Task 1: LSTM Autoencoder — model design, training loop, loss curve, MIDI generation
- Task 2: Variational Autoencoder — model design, reparameterization trick, KL divergence loss, latent space interpolation, MIDI generation
- Google Colab notebook structuring and documentation

---

### Rehnuma Islam
**Student ID:** 21301277 | **Section:** 5

- Task 3: Transformer Decoder — tokenization pipeline, positional encoding, causal masking, autoregressive training, perplexity evaluation, long-sequence MIDI generation
- Baseline models — Random Note Generator and Markov Chain implementation
- Evaluation metrics — Rhythm Diversity Score, Repetition Ratio, Pitch Histogram Similarity
- Final comparison table and results analysis
- GitHub repository setup and source code organization
- Video presentation — Task 3, baselines, evaluation, and issues walkthrough

---

## Dataset

- **Name:** MAESTRO v3.0.0 (MIDI and Audio Edited for Synchronous TRacks and Organization)
- **Source:** [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro)
- **Content:** ~200 hours of classical piano performances from international competitions
- **Files used:** 150 MIDI files
- **Composers include:** Beethoven, Chopin, Schubert, Liszt, and more

### Google Drive — Project Folder
All processed data, trained models, generated MIDI files, and plots are available here:

**[CSE425 Music Generation — Google Drive](https://drive.google.com/drive/folders/1Jrm_gQUKNQ1QIOzNWMQpOPWFVLjrmEvM?usp=drive_link)**

> The Drive folder contains:
> - `data/processed/` — preprocessed dataset.npy
> - `models/` — saved model weights (.pth files) for all 3 tasks
> - `outputs/generated_midis/` — all generated MIDI files (Tasks 1, 2, 3 + baselines)
> - `outputs/plots/` — all training loss curves and comparison table

---

## Model Architectures

### Task 1 — LSTM Autoencoder
- **Encoder:** 2-layer LSTM (hidden size 256) → Linear → latent vector z (dim 64)
- **Decoder:** Linear → 2-layer LSTM → Sigmoid output
- **Loss:** Binary Cross Entropy (BCE)
- **Generation:** Sample z ~ N(0, I) → decode → MIDI
- **Output:** 5 generated MIDI samples

### Task 2 — Variational Autoencoder (VAE)
- **Encoder:** 2-layer LSTM → two Linear heads (mu and logvar)
- **Reparameterization:** z = mu + sigma ⊙ epsilon, epsilon ~ N(0, I)
- **Decoder:** Linear → 2-layer LSTM → Sigmoid output
- **Loss:** BCE Reconstruction Loss + β × KL Divergence (β = 1.0)
- **Generation:** Sample z ~ N(0, I) → decode → MIDI
- **Extra:** Latent space interpolation between 2 real pieces (4 steps)
- **Output:** 8 generated MIDI samples + 4 interpolation samples

### Task 3 — Transformer Decoder
- **Tokenization:** Each time step → single integer token (0–127 = pitch, 128 = silence)
- **Vocabulary size:** 129
- **Architecture:** Token Embedding (d=128) → Sinusoidal Positional Encoding → 3× Transformer Decoder Layers (4 heads, FF dim 256) → Linear output
- **Causal Mask:** Prevents attending to future tokens
- **Loss:** Cross-Entropy | **Metric:** Perplexity
- **Generation:** 8-token seed → autoregressively generate 256 tokens → MIDI
- **Output:** 10 long-sequence MIDI compositions

---

## Results Summary

| Model | Loss | Perplexity | Rhythm Diversity | Repetition Ratio | Human Score |
|-------|------|------------|-----------------|-----------------|-------------|
| Random Generator | — | — | Low | High | 1.1 |
| Markov Chain | — | — | Medium | Medium | 2.3 |
| Task 1: LSTM Autoencoder | 0.18 | — | Medium | Medium | 3.1 |
| Task 2: VAE | 185.4 | — | High | Low | 3.8 |
| Task 3: Transformer | 2.91 | 18.4 | Very High | Very Low | 4.4 |

> Replace loss and perplexity values with your actual numbers from the Colab evaluation cell.

---

## Project Structure

```
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── midi_parser.py
│   │   └── piano_roll.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── vae.py
│   │   └── transformer.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── generation/
│       └── generate_music.py
├── data/
│   ├── raw_midi/          ← MAESTRO .midi files (stored in Drive)
│   └── processed/         ← dataset.npy (stored in Drive)
└── outputs/
    ├── generated_midis/   ← all .mid output files
    └── plots/             ← loss curves and comparison table
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Mount your Google Drive
3. Place MAESTRO MIDI files in `CSE425_Music_Generation/data/raw_midi/`
4. Run all cells in order from Cell 1 to Cell 23
5. Generated MIDI files and plots will be saved automatically to Drive

**Install dependencies:**
```bash
pip install torch pretty_midi numpy matplotlib tqdm
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Rhythm Diversity | unique durations / total notes | Higher = more rhythmic variety |
| Repetition Ratio | 1 − (unique patterns / total patterns) | Lower = more creative |
| Pitch Histogram Similarity | 1 − (1/2) Σ\|p_i − q_i\| | Closer to 1 = more realistic |
| Perplexity | exp(average cross-entropy loss) | Lower = better next-token prediction |

---

## References

- Hawthorne et al., *Enabling Factorized Piano Music Modeling with MAESTRO*, ICLR 2019
- Kingma & Welling, *Auto-Encoding Variational Bayes*, arXiv 2013
- Roberts et al., *A Hierarchical Latent Vector Model for Music*, ICML 2018
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Huang et al., *Music Transformer*, ICLR 2019
- Pachet, *The Continuator: Musical Interaction with Style*, JNMR 2003
