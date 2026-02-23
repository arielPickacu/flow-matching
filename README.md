# Scalable Interpolant Transformer (SiT) for Text-to-Image

# Generation

This repository contains a PyTorch implementation of a Scalable Interpolant Transformer (SiT)
trained on the MS-COCO 2017 dataset. This project leverages Flow Matching to learn the
velocity field of the interpolant, transitioning from Gaussian noise to data latents.

<p align="center"> <img src="images_of_vae/human_test_00000.png" width="200" /> <img
src="images_of_vae/human_test_00001.png" width="200" /> <img
src="images_of_vae/human_test_00003.png" width="200" /> <img
src="images_of_vae/human_test_00004.png" width="200" /> </p>

## 🚀 Project Overview

```
Backbone: SiT-B/2 (Base Transformer, Patch Size 2)
Framework: Flow Matching / Velocity Prediction
Interpolant: Linear (Optimal Transport)
Conditioning: Text-to-Image via CLIP-ViT-B/32 (pooler output)
Compression: Latent space training via pre-trained SD-VAE (8x downsampling)
Dataset: MS-COCO 2017
```
## 🏗 Technical Performance

The model achieved an FID score of 61 after approximately 200 epochs (~370,000 steps) on a
single RTX 4090. This score represents a strong baseline for text-conditional generation on the
highly complex, multi-object COCO dataset given the compute and time constraints.

## 📥 Setup and Data

1. Dependencies

Install the required packages using the provided requirements file:

```
pip install -r requirements.txt
```
_(Note: Ensure you have_ pytorch-fid _installed for benchmarking)._

2. High-Speed Data Download

You can download and extract the COCO dataset using parallel streams via aria2c. Create a
file named setup_coco.sh and run it:


```bash
set -e
BASE_DIR="/workspace/coco2017"
CONNECTIONS=
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"
echo "Downloading Train Images..."
aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -o train2017.zip [http://images.cocoda
7z x train2017.zip -o"$BASE_DIR" -mmt="$CONNECTIONS"
echo "Downloading Validation Images..."
aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -o val2017.zip [http://images.cocodata
unzip -q val2017.zip -d "$BASE_DIR"
echo "Downloading Annotations..."
aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -o annotations_trainval2017.zip [http:
unzip -q annotations_trainval2017.zip -d "$BASE_DIR/annotations"
```
## 🖼 Inference & Generation

The primary latent flow-matching code is located in the with_vae/ directory. The generator
supports two distinct modes:

```
FID Mode: Uses a lower CFG scale (1.5) to maintain distribution diversity for benchmarking
against the COCO validation set.
Human Mode: Uses a higher CFG scale (7.0) and SDE (Stochastic) sampling for sharper,
more detailed visual results.
```
Run the generator:

```
cd with_vae
python generate.py
```
## 📜 Acknowledgments & Attributions

This project is built upon the foundational work of the generative AI research community:

Core Architecture & Methodology

```
SiT (Scalable Interpolant Transformers): Based on SiT: Exploring Flow and Diffusion-based
Generative Models with Scalable Interpolant Transformers (Ma et al., 2024).
DiT (Diffusion Transformers): Based on Scalable Diffusion Models with Transformers
(Peebles & Xie, 2023).
Flow Matching: Utilizing principles from Flow Matching for Generative Modeling (Lipman et
al., 2023).
```
Component Acknowledgments


```
Variational Autoencoder (VAE): We utilize the pre-trained sd-vae-ft-mse provided by
Stability AI for latent space encoding/decoding.
Text Encoder: Text conditioning is performed using the CLIP-ViT-B/32 model developed by
OpenAI.
```
Dataset

```
MS-COCO: This model is trained on the Microsoft COCO: Common Objects in Context
dataset. Annotations are provided by the COCO Consortium under a CC BY 4.0 license.
Citation: Lin, T. Y., et al. (2014). "Microsoft COCO: Common Objects in Context."
```
## ⚖ License

The code, configuration files, and trained model weights in this repository are released under
the MIT License.

Copyright (c) 2026 Ariel Mazor

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


