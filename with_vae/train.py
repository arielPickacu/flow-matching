#set up
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import logging
torch.backends.cudnn.benchmark = True
from model import SiT,EMA
import os
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import LambdaLR
import math
import random
def get_lr_scheduler(optimizer, total_steps, warmup_steps=1000):
    def lr_lambda(current_step):
        # 1. Linear Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Cosine Annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

class CachedCOCODataset(torch.utils.data.Dataset):
    def __init__(self, latents_file, clip_file):
        """
        A highly optimized dataset that reads pre-computed VAE latents 
        and CLIP embeddings from monolithic dictionary .pt files.
        """
        print("Loading precomputed VAE latents into memory... (this might take a moment)")
        # Note: weights_only=False is used because we are loading a Python dictionary, 
        # not a pure model state_dict.
        self.latents_dict = torch.load(latents_file, weights_only=False)
        
        print("Loading precomputed CLIP embeddings into memory...")
        self.clip_dict = torch.load(clip_file, weights_only=False)
        
        # 1. Safety Check: Find image IDs that exist in BOTH dictionaries
        # This prevents crashes if an image failed to process in one of the scripts
        latent_ids = set(self.latents_dict.keys())
        clip_ids = set(self.clip_dict.keys())
        
        # 2. Store the valid intersecting IDs as a sorted list for deterministic ordering
        self.valid_ids = sorted(list(latent_ids.intersection(clip_ids)))
        print(f"Found {len(self.valid_ids)} perfectly matched image-text pairs ready for training!")
        
    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        # Retrieve the aligned COCO image ID
        img_id = self.valid_ids[idx]
        
        # 1. Get the visual representation (Shape: 4, 32, 32)
        latent = self.latents_dict[img_id]
        
        # 2. Get the text representations
        # Because COCO has ~5 captions per image, your CLIP script stored them as a stacked tensor.
        # Shape is usually (5, 512) for CLIP-ViT-Base.
        clip_embeds = self.clip_dict[img_id]
        
        # 3. Dynamic Caption Sampling
        # Randomly select one of the ~5 captions for this specific epoch.
        # This acts as essential data augmentation so the model doesn't just memorize 
        # the very first caption for every image.
        caption_idx = random.randint(0, len(clip_embeds) - 1)
        single_clip_embed = clip_embeds[caption_idx]
        
        return latent, single_clip_embed

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = "/workspace/flow_matching_with_text_n.tar"
    ema_ckpt_path = "/workspace/ema_flow_matching_text_n.pt"

    # 1. Initialize SiT-B/2 Architecture
    # Base params: hidden_size=768, num_heads=12, depth=12
    print("Initializing SiT-B/2 model...")
    model = SiT(input_size=32, in_channels=4, text_size=512, depth=12, hidden_size=768, patch_size=2, num_heads=12, learn_sigma=False).to(device)

    # 2. Initialize Optimizer and Scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_training_steps = 1849 * 200 # Adjust based on your dataset size/epochs
    scheduler = get_lr_scheduler(optim, total_steps=total_training_steps, warmup_steps=500)

    # 3. Create the EMA model (starts as a perfect copy of the fresh model)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.requires_grad_(False)

    # 4. Attempt to load Main Model Checkpoint
    if os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}. Resuming training...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        print(f"No checkpoint found at {ckpt_path}. Starting fresh.")

    model.train()

    # 5. Attempt to load EMA Model Checkpoint
    if os.path.exists(ema_ckpt_path):
        print(f"Found EMA checkpoint at {ema_ckpt_path}. Loading...")
        ema_model.load_state_dict(torch.load(ema_ckpt_path, map_location=device, weights_only=True))
    else:
        print("No EMA checkpoint found. Using fresh EMA model.")

    # Note: EMA models should permanently remain in eval() mode
    ema_model.eval()

    return model, ema_model, optim, scheduler

def save(model,ema_model,optim):
    torch.save({
            'model_state_dict': model._orig_mod.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            }, "/workspace/flow_matching_with_text_n.tar")
    torch.save(ema_model.state_dict(),"/workspace/ema_flow_matching_text_n.pt" )

def get_loader(root = "train2017",annFile = "annotations/annotations/captions_train2017.json"):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CachedCOCODataset(latents_file="/workspace/coco_vae_latents.pt", clip_file="/workspace/coco_clip_text_embeddings.pt")

    loader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=8, 
        persistent_workers=True, 
        pin_memory=True
    )
    return loader



def train(model,ema_model,optim,loader,loss,scheduler,everysteps_save = 2000,device='cuda',p_uncond = 0.1,training_steps = 40):
    ema = EMA(0.995)
    scaler = torch.amp.GradScaler(device=device)
    losses = []
    for epoch in range(training_steps):
        pbar = tqdm.tqdm(loader)
        for i, (x1, text_emb) in enumerate(pbar):
            # Ensure data is on device and in float32
            x1 = x1.to(device,non_blocking=True).float()
            text_emb = text_emb.to(device, non_blocking=True)

            # 2. Classifier-Free Dropout (per-sample)
            # Create a mask: 10% True (drop), 90% False (keep)
            drop_mask = (torch.rand(x1.size(0), 1, device=device) < p_uncond)
            # Replace text with zeros (null) where mask is True
            text_emb = torch.where(drop_mask, torch.zeros_like(text_emb), text_emb)

            # 3. Flow-matching logic (Float32 for precision)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=device)
            xt = torch.lerp(x0, x1, t.view(-1, 1, 1, 1))
            target = x1 - x0

            # 4. Forward & Backward with Mixed Precision
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(xt, t, text_emb)
                total_loss = loss(pred, target)

            optim.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            ema.step_ema(ema_model, model._orig_mod)
            losses.append(total_loss.item())
            pbar.set_postfix({"loss": total_loss.item()})
            if (i%everysteps_save == 0):
                save(model,ema_model,optim)
    save(model,ema_model,optim)

if True:
    model,ema_model,optim,scheduler = load()
    model = torch.compile(model)
    loader = get_loader()
    loss = torch.nn.MSELoss()
    # --- Inside your main block ---
    train(model,ema_model,optim,loader,loss,scheduler,everysteps_save = 1000,device='cuda',p_uncond = 0.1,training_steps = 160)