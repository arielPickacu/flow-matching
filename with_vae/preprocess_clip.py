import os
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGES_DIR = "/workspace/coco2017/train2017" # Point this to your extracted COCO images
OUTPUT_FILE = "coco_vae_latents.pt"
BATCH_SIZE = 64 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def main():
    print(f"Loading VAE model on {DEVICE}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.requires_grad_(False)
    vae.eval()
    
    # Dynamically grab the correct scaling factor (usually ~0.18215)
    scaling_factor = vae.config.scaling_factor

    print(f"Scanning images in {IMAGES_DIR}...")
    image_filenames = [f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")]
    print(f"Found {len(image_filenames)} images.")

    # The VAE expects inputs normalized between -1 and 1
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dictionary to map image_id to its latent tensor
    latents_dict = {}
    
    # Process in batches for speed
    for i in tqdm(range(0, len(image_filenames), BATCH_SIZE), desc="Processing Images"):
        batch_filenames = image_filenames[i:i + BATCH_SIZE]
        batch_images = []
        batch_ids = []
        
        for filename in batch_filenames:
            # COCO filenames are zero-padded IDs (e.g., '000000391895.jpg')
            # Stripping the extension and converting to int matches the JSON image_id
            img_id = int(filename.split('.')[0])
            batch_ids.append(img_id)
            
            # Load, convert to RGB (to avoid grayscale channel errors), and transform
            img_path = os.path.join(IMAGES_DIR, filename)
            img = Image.open(img_path).convert("RGB")
            batch_images.append(transform(img))
            
        # Stack into a single tensor: (B, 3, 256, 256) and push to GPU
        inputs = torch.stack(batch_images).to(DEVICE)
        
        with torch.no_grad():
            # Encode to distribution, sample, and scale
            latent_dist = vae.encode(inputs).latent_dist
            latents = latent_dist.sample() * scaling_factor
            
            # Move off GPU to prevent RAM explosion during dictionary storage
            latents = latents.cpu()
            
        # Store in dictionary (Shape per image: 4, 32, 32)
        for img_id, latent in zip(batch_ids, latents):
            latents_dict[img_id] = latent

    print(f"Saving precomputed latents to {OUTPUT_FILE}...")
    torch.save(latents_dict, OUTPUT_FILE)
    print("Done! Both your visual and text representations are now cached.")

if __name__ == "__main__":
    main()