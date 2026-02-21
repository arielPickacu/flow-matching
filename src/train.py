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
os.chdir("/workspace/coco2017")
print(os.getcwd())

def get_lr_scheduler(optimizer, total_steps, warmup_steps=1000):
    def lr_lambda(current_step):
        # 1. Linear Warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Cosine Annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

class CocoPrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, embeddings_path, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        # Load the 240MB dictionary into RAM once
        print("Loading precomputed embeddings into RAM...")
        self.embs_dict = torch.load(embeddings_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, _ = self.dataset[idx]
            
            # Get the COCO image ID to look up the text embedding
            img_id = self.dataset.ids[idx]

            if self.transform:
                img = self.transform(img)

            # Instantly grab the pre-calculated [512] tensor
            text_emb = self.embs_dict[img_id]

            return img, text_emb

        except Exception as e:
            # If an image is broken, skip to the next one
            return self.__getitem__((idx + 1) % len(self))

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiT(input_size=64,in_channels=3, text_size=512, depth=12, hidden_size=512, patch_size=4, num_heads=16, learn_sigma=False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.to(device)
    checkpoint = torch.load("/models/flow_matching_with_text_n.tar", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    total_training_steps = 1849 * 200
    scheduler = get_lr_scheduler(optim, total_steps=total_training_steps, warmup_steps=500)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    model.train()

    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
    ema_model.load_state_dict(torch.load("/models/ema_flow_matching_text_n.pt", weights_only=True))
    ema_model.eval()
    ema_model.train()
    return model,ema_model,optim,scheduler

def save(model,ema_model,optim):
    torch.save({
            'model_state_dict': model._orig_mod.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            }, "/workspace/flow_matching_with_text_n.tar")
    torch.save(ema_model.state_dict(),"/models/ema_flow_matching_text_n.pt" )

def get_loader(root = "train2017",annFile = "annotations/annotations/captions_train2017.json"):
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data = CocoDetection(root=root, annFile=annFile)

    dataset = CocoPrecomputedDataset(
        dataset=data, 
        embeddings_path="coco_embeddings.pt", 
        transform=transform
    )

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

            if(i % 100 == 0):
                # EMA and Logging
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
    train(model,ema_model,optim,loader,loss,scheduler,everysteps_save = 1000,device='cuda',p_uncond = 0.1,training_steps = 60)