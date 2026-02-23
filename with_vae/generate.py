import torch
import numpy as np
import os
import json
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from model import SiT

@torch.no_grad()
def run_sampler_sde_batch(model, z, text_embeds, num_steps, cfg_scale, device, gamma=0.1):
    dt = 1.0 / num_steps
    # Expand z for CFG (batch_size * 2)
    # text_embeds is already (batch_size * 2, 512)
    
    for step in range(num_steps):
        t_val = step / num_steps
        t = torch.full((z.shape[0],), t_val, device=device)
        
        # CFG Prep: Duplicate latents and timesteps
        z_in = torch.cat([z, z], dim=0) 
        t_in = torch.cat([t, t], dim=0) 
        
        # 1. Predict Velocity
        v_pred = model(z_in, t_in, text_embeds)
        v_uncond, v_cond = v_pred.chunk(2)
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # 2. ODE Step
        z = z + v * dt
        
        # 3. SDE Noise (Langevin)
        if 0.1 < t_val < 0.8:
            noise = torch.randn_like(z)
            z = z + (gamma * np.sqrt(dt)) * noise
            
    return z

def batch_generate_for_fid(
    coco_json_path, 
    output_dir="/workspace/fid_samples",
    batch_size=8,
    cfg_scale=4.5, 
    num_steps=50,
    num_images=5000,
    seed=42
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed) # Set initial global seed

    # 1. Load Prompts from COCO JSON
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    # Extract unique prompts (COCO has multiple per image; we just need one per ID for FID)
    all_prompts = [ann['caption'] for ann in data['annotations']][:num_images]

    # 2. Load Models
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    
    model = SiT(input_size=32, in_channels=4, text_size=512, depth=12, hidden_size=768, 
                patch_size=2, num_heads=12, learn_sigma=False).to(device).eval()
    model.load_state_dict(torch.load("/workspace/ema_flow_matching_text_n.pt", map_location=device))

    # 3. Batch Processing Loop
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i : i + batch_size]
        curr_batch_size = len(batch_prompts)
        
        # Encode Text (Conditional + Unconditional)
        # We prepare [null, null, null..., prompt1, prompt2, prompt3...]
        prompts_to_encode = [""] * curr_batch_size + batch_prompts
        inputs = tokenizer(prompts_to_encode, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            text_embeds = text_encoder(inputs.input_ids).pooler_output # Shape: (batch_size*2, 512)
            
            # Initialize Latent Noise
            z = torch.randn(curr_batch_size, 4, 32, 32, device=device)
            
            # Sample using SDE
            z = run_sampler_sde_batch(model, z, text_embeds, num_steps, cfg_scale, device)

            # Decode
            z = z / vae.config.scaling_factor
            imgs = vae.decode(z).sample
            
            imgs = (imgs / 2 + 0.5).clamp(0, 1).detach().float().cpu().permute(0, 2, 3, 1).numpy()

        # Save individual images
        for j, img_np in enumerate(imgs):
            img = Image.fromarray((img_np * 255).astype(np.uint8))
            img_idx = i + j
            img.save(os.path.join(output_dir, f"sample_{img_idx:05d}.png"))

if __name__ == "__main__":
    # Point this to your COCO Val annotations file
    batch_generate_for_fid(
        coco_json_path="/workspace/coco2017/annotations/annotations/captions_val2017.json",
        num_images=5000, 
        batch_size=16 
    )