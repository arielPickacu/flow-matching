import os
from huggingface_hub import hf_hub_download, snapshot_download

# --- CONFIGURATION ---
# Replace these with your actual Hugging Face repository names!
MODEL_REPO = "arielPickacu/Sit_Recreation"
# ---------------------

def download_weights():
    print("Downloading model weights...")
    os.makedirs("models", exist_ok=True)
    
    # Downloads the specific model.pth file
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, 
        filename="ema_flow_matching_text_n.pt", 
        local_dir="models" # Saves it to a local folder named 'models'
    )
    ema_model_path = hf_hub_download(
        repo_id=MODEL_REPO, 
        filename="flow_matching_with_text_n.tar", 
        local_dir="models" # Saves it to a local folder named 'models'
    )
    print(f"Model downloaded to: {model_path}")
    print(f"EmaModel downloaded to: {ema_model_path}")

if __name__ == "__main__":
    download_weights()
    print("All assets downloaded successfully!")