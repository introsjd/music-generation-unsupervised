# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425/EEE474 Neural Networks — Spring 2026

## Overview
Three unsupervised deep learning models for symbolic music generation
trained on the MAESTRO classical piano MIDI dataset.

## Tasks
| Task | Model | Metric |
|------|-------|--------|
| Task 1 | LSTM Autoencoder | Reconstruction Loss |
| Task 2 | Variational Autoencoder (VAE) | ELBO + KL Divergence |
| Task 3 | Transformer Decoder | Perplexity |

## Dataset
- MAESTRO v3.0.0 — Classical piano MIDI
- 150 files, ~8000 segments, 80/20 train/val split

## Project Structure
    src/
      config.py
      preprocessing/
        midi_parser.py
        piano_roll.py
      models/
        autoencoder.py
        vae.py
        transformer.py
      evaluation/
        metrics.py
      generation/
        generate_music.py
    outputs/
      generated_midis/
      plots/
    requirements.txt

## Results
| Model | Rhythm Diversity | Repetition Ratio | Human Score |
|-------|-----------------|-----------------|-------------|
| Random Generator | Low | High | 1.1 |
| Markov Chain | Medium | Medium | 2.3 |
| Task 1: LSTM AE | Medium | Medium | 3.1 |
| Task 2: VAE | High | Low | 3.8 |
| Task 3: Transformer | Very High | Very Low | 4.4 |
