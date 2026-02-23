import torch
import os
from pytorch_fid import fid_score

def calculate_model_fid(real_images_path, generated_images_path, batch_size=64, device='cuda'):
    # Check for consistency
    real_imgs = os.listdir(real_images_path)
    gen_imgs = os.listdir(generated_images_path)
    
    print(f"Starting FID calculation...")
    print(f"Real folder: {len(real_imgs)} files")
    print(f"Gen folder: {len(gen_imgs)} files")

    # Setting num_workers=0 helps debug resizing/corrupt file errors
    score = fid_score.calculate_fid_given_paths(
        paths=[real_images_path, generated_images_path],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0  # Change from default to 0 to avoid the Resize Storage error
    )
    return score


if __name__ == "__main__":
    # Standard COCO Val path and your output path
    REAL_DIR = "/workspace/coco_standardized_256"
    GEN_DIR = "/workspace/fid_samples"

    try:
        fid = calculate_model_fid(REAL_DIR, GEN_DIR)
        print(f"--- FINAL RESULTS ---")
        print(f"Current FID Score: {fid:.4f}")
    except Exception as e:
        print(f"FID Calculation Failed: {e}")