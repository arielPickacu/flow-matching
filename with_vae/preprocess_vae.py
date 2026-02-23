import os
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGES_DIR = "/workspace/coco2017/train2017" # Point this to your extracted COCO images
OUTPUT_FILE = "coco_vae_latents.pt"
BATCH_SIZE = 64 s
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8 # Adjust this based on your CPU cores (usually 4 to 8 is good)
# ---------------------

# 1. Create a proper PyTorch Dataset
class COCODataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_id = int(filename.split('.')[0])
        img_path = os.path.join(self.img_dir, filename)
        
        # Open and apply transforms
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        return img, img_id

def main():
    # Quick sanity check for CUDA
    if DEVICE == "cpu":
        print("WARNING: CUDA is not available! PyTorch is defaulting to CPU.")
        
    print(f"Loading VAE model on {DEVICE}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae.requires_grad_(False)
    vae.eval()
    
    scaling_factor = vae.config.scaling_factor

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"Scanning images in {IMAGES_DIR}...")
    
    # 2. Instantiate Dataset and DataLoader
    dataset = COCODataset(IMAGES_DIR, transform)
    
    # The DataLoader is the magic fix. 
    # pin_memory=True speeds up the transfer from CPU RAM to GPU VRAM
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )

    print(f"Found {len(dataset)} images.")

    latents_dict = {}
    
    # 3. Iterate over the DataLoader instead of manual loops
    for inputs, batch_ids in tqdm(dataloader, desc="Processing Images"):
        
        # Move the pre-processed batch to the GPU
        inputs = inputs.to(DEVICE)
        
        with torch.no_grad():
            latent_dist = vae.encode(inputs).latent_dist
            latents = latent_dist.sample() * scaling_factor
            latents = latents.cpu()
            
        # Store in dictionary 
        # .item() converts the tensor ID back to a standard Python integer
        for img_id, latent in zip(batch_ids, latents):
            latents_dict[img_id.item()] = latent

    print(f"Saving precomputed latents to {OUTPUT_FILE}...")
    torch.save(latents_dict, OUTPUT_FILE)
    print("Done! Both your visual and text representations are now cached.")

if __name__ == "__main__":
    main()