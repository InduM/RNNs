# Rewriting the README.md after kernel reset

readme_content = """
# Multi-Task RNN for Educational NLP Tasks

This project implements a simple multi-task Recurrent Neural Network (RNN) in PyTorch that performs two distinct NLP tasks using a shared architecture:

1. 📚 Educational Text Classification — Classifies short educational snippets into categories: `Math`, `Science`, or `History`.
2. ✍️ Next Word Generation — Given a prompt of at least 10 words, the model generates the next 20 words, simulating intelligent auto-completion.

---

## 🧠 Model Architecture

A shared embedding + RNN encoder supports both tasks:
- For classification: uses the final hidden state with a linear layer for category prediction.
- For generation: applies a linear decoder over each timestep to predict the next word.

---

## 🔧 Setup

### Requirements

- Python 3.8+
- PyTorch 2.x+
- transformers
- scikit-learn
- tqdm

### Install Dependencies

```bash
pip install torch transformers scikit-learn tqdm
