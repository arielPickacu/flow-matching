# SiT with CLIP Text Embeddings

This repository contains the training and model architecture code for a Scalable Interpolant Transformer (SiT) conditioned on CLIP text embeddings, trained on the COCO dataset.

Because the trained model weights and precomputed datasets are too large for GitHub, they are hosted on Hugging Face.

## 📦 Model and Dataset Links
* **Trained Model Weights:** [Link to your Hugging Face Model Repo here]
* **Precomputed Text Embeddings:** [Link to your Hugging Face Dataset Repo here]

## 🛠️ Setup and Installation

### 1. Install the Dependencies
First, clone the repository and install the required Python libraries:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
```
### 2. Download Model Weights & Text Embeddings
To automatically fetch the trained `.pth` weights and the precomputed CLIP embeddings from Hugging Face, run the included download script:
```bash
python download_assets.py
```

3. Download the COCO 2017 Dataset (Fast Parallel Download)
To download the COCO 2017 dataset quickly using parallel connections, make sure your system has aria2 and 7zip installed:

```bash
sudo apt-get update
sudo apt-get install aria2 p7zip-full unzip
```

Next, run the included Bash script to rapidly download and extract the dataset to the correct directory (/workspace/coco2017):


```bash
# Make the script executable
chmod +x download_coco.sh

# Run the script
./download_coco.sh
```

clip preprocess:
python preprocess_clip.py
🚀 Training
Once the dataset and dependencies are set up, you can start the training process:
```bash
python train.py
```
generate:
you can generate in notebook. (run the import part, download the model with the correct path and run the generate part)
📜 License
MIT