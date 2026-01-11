# MNIST 0 vs 1 Classifier (PyTorch)

A simple binary classifier that learns to distinguish handwritten **0** vs **1** from MNIST.

## What it does
- Downloads MNIST automatically
- Filters dataset to only digits 0 and 1
- Binarizes pixels (0/1) with a threshold
- Trains a small MLP and prints accuracy + sample predictions

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
