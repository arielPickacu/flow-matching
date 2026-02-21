import os
import json
import torch
from transformers import CLIPProcessor, CLIPTextModel
from tqdm import tqdm

# --- CONFIGURATION ---
ANNOTATIONS_PATH = "/workspace/coco2017/annotations/captions_train2017.json"
OUTPUT_FILE = "coco_clip_text_embeddings.pt"
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------

def main():
    print(f"Loading CLIP model on {DEVICE}...")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading COCO annotations...")
    with open(ANNOTATIONS_PATH, 'r') as f:
        coco_data = json.load(f)
        
    annotations = coco_data['annotations']
    print(f"Found {len(annotations)} captions.")

    # Dictionary to map image_id to its text embeddings
    # Note: COCO has ~5 captions per image, we will store them in a list per image_id
    embeddings_dict = {}
    
    # Process in batches for speed
    for i in tqdm(range(0, len(annotations), BATCH_SIZE), desc="Processing Captions"):
        batch_anns = annotations[i:i + BATCH_SIZE]
        texts = [ann['caption'] for ann in batch_anns]
        image_ids = [ann['image_id'] for ann in batch_anns]
        
        # Tokenize and push to GPU
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        
        with torch.no_grad():
            # Get the pooled output (e.g., 512-dimensional vector)
            outputs = model(**inputs)
            text_embeds = outputs.pooler_output.cpu() 
            
        # Store in dictionary
        for img_id, emb in zip(image_ids, text_embeds):
            if img_id not in embeddings_dict:
                embeddings_dict[img_id] = []
            embeddings_dict[img_id].append(emb)

    # Stack the lists into tensors for each image
    for img_id in embeddings_dict:
        embeddings_dict[img_id] = torch.stack(embeddings_dict[img_id])

    print(f"Saving precomputed embeddings to {OUTPUT_FILE}...")
    torch.save(embeddings_dict, OUTPUT_FILE)
    print("Done! You can now upload this file to your Hugging Face Dataset Hub.")

if __name__ == "__main__":
    main()