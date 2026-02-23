import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

def prepare_fid_folder(src_dir, dst_dir, size=256):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Standardize transform: Resize smaller side to 256, then center crop
    transform = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.Lambda(lambda x: x.convert('RGB')) # Force 3 channels (removes Alpha or Grayscale)
    ])

    print(f"Standardizing {len(files)} images to {size}x{size}...")
    for f in tqdm(files):
        try:
            with Image.open(os.path.join(src_dir, f)) as img:
                standard_img = transform(img)
                standard_img.save(os.path.join(dst_dir, f))
        except Exception as e:
            print(f"Skipping {f}: {e}")

# --- EXECUTE THIS FIRST ---
REAL_SRC = "/workspace/coco2017/val2017"
REAL_DST = "/workspace/coco_standardized_256"

prepare_fid_folder(REAL_SRC, REAL_DST)