import os
from huggingface_hub import hf_hub_download, snapshot_download

# --- CONFIGURATION ---
# Replace these with your actual Hugging Face repository names!
MODEL_REPO = "your-username/sit-cifar10-model"
DATA_REPO = "your-username/coco-clip-embeddings"
# ---------------------

def download_weights():
    print("Downloading model weights...")
    os.makedirs("models", exist_ok=True)
    
    # Downloads the specific model.pth file
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, 
        filename="model.pth", 
        local_dir="models" # Saves it to a local folder named 'models'
    )
    print(f"Model downloaded to: {model_path}")

def download_dataset():
    print("Downloading precomputed CLIP dataset...")
    os.makedirs("data", exist_ok=True)
    
    # Downloads the entire dataset repository (e.g., your .pt or .npy files)
    dataset_path = snapshot_download(
        repo_id=DATA_REPO, 
        repo_type="dataset",
        local_dir="data"
    )
    print(f"Dataset downloaded to: {dataset_path}")

if __name__ == "__main__":
    download_weights()
    download_dataset()
    print("All assets downloaded successfully!")