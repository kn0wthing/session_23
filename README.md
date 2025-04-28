# Vision-Language Model Training Framework

A framework for training Vision-Language Models (VLMs) on CIFAR-10 using SigLIP embeddings and Phi-2.

## Overview

This project builds a Vision-Language Model through a multi-stage approach:

1. **SigLIP Training**: Creates efficient image embeddings aligned with text
2. **Phi-2 VLM Training**: Fine-tunes Phi-2 with QLoRA using SigLIP embeddings
3. **Inference**: Analyzes images with detailed text descriptions

## Installation

```bash
git clone https://github.com/kn0wthing/vision-language-model-training.git
cd vision-language-model-training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

- torch>=2.0.0
- transformers>=4.36.0
- peft>=0.7.0
- bitsandbytes>=0.41.1
- wandb
- tensorboard
- torchvision

## Dataset Format

The framework expects a CSV with the following structure:
- `Dataset_Index`: Index in CIFAR-10
- `Q1-Q5`: Questions about each image
- `A1-A5`: Text answers/descriptions

## Training Pipeline

### 1. SigLIP Training

```bash
python train_siglip.py \
  --input-csv path/to/dataset.csv \
  --batch-size 32 \
  --num-epochs 10
```

### 2. Phi-2 VLM Training

```bash
python train_phi2_vlm.py \
  --input-csv path/to/dataset.csv \
  --siglip-checkpoint path/to/siglip_model/final_model.pt \
  --batch-size 16 \
  --num-epochs 3
```

## Model Architecture

- **SigLIP**: ResNet50 backbone with projection to 512-dimensional space
- **Phi-2 VLM**: Microsoft's Phi-2 with QLoRA and SigLIP image embeddings

## Optimization Techniques

- 4-bit quantization
- Gradient checkpointing
- QLoRA fine-tuning
- Mixed precision training
- Flash Attention 2 (when available)
- Rank-stabilized LoRA

## Training Results

Below are screenshots from the training process:

### SigLIP Training

<img src="assets/siglip_train_1.png" width="800" alt="SigLIP Training Loss">

### Phi-2 VLM Training

<img src="imgs/Screenshot 2025-04-26 at 1.26.27 AM.png" width="800" alt="Phi-2 VLM Training Start">
<img src="imgs/Screenshot 2025-04-26 at 1.27.27 AM.png" width="800" alt="Phi-2 VLM Training Progress">
<img src="imgs/Screenshot 2025-04-26 at 1.28.00 AM.png" width="800" alt="Phi-2 VLM Training Completion">



