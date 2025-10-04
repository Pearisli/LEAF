# <img src="assets/images/leaf.png" alt="leaf" style="height:1em; vertical-align:bottom;"/> LEAF: Latent Diffusion with Efficient Encoder Distillation for Aligned Features in Medical Image Segmentation

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://leafseg.github.io/leaf/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.18214)
[![Segmentation Model](https://img.shields.io/badge/🤗%20Segmentation-Model-green)](https://huggingface.co/pearisli/LEAF-QaTa-COV19)

## LEAF Framework
<img src="assets/images/main_pipeline.jpg" alt="leaf_pipeline" style=" vertical-align:bottom;"/>

## Performance
<img src="assets/images/performance.png" alt="leaf_performance" style=" vertical-align:bottom;"/>

## Ablation
<img src="assets/images/ablation.png" alt="leaf_ablation" style=" vertical-align:bottom;"/>

## Update
- **Oct 2025**: Release [LEAF-QaTa-COV19](https://huggingface.co/pearisli/LEAF-QaTa-COV19) model on huggingface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lispear/LEAF.git
cd LEAF-master
```

2. Install dependencies (requires conda):
```bash
conda create -n leaf python=3.11.11 -y
conda activate leaf
pip install -r requirements.txt 
```

## Training

1. Create assets directory:
```
mkdir assets
cd assets
```

2. Prepare pre-trained models:
- Download [U-Net](https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip) and [VAE](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip) and **extract weights**
```bash
unzip kl-f8.zip -d vae
unzip lsun_churches.zip -d unet
cd ..
python extract_weights.py
```

3. Run training script:
```bash
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision 'no' \
    --dynamo_backend 'no' \
    train.py --config config.yaml
```
