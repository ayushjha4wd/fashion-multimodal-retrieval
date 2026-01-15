import os, json, yaml, faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.clip_loader import load_clip
from models.image_encoder import encode_image
from indexer.store_vectors import build_index

ATTRIBUTES = {
    "clothing": ["shirt", "t-shirt", "jacket", "blazer", "raincoat"],
    "color": ["red", "blue", "yellow", "black", "white"],
    "context": ["office", "park", "street", "home"],
    "style": ["formal", "casual"]
}

def main():
    cfg = yaml.safe_load(open("config/model_config.yaml"))
    model, preprocess, _ = load_clip(
        cfg["clip_model"], cfg["pretrained"], cfg["device"]
    )

    image_dir = "data/raw/images"
    image_ids = []
    vectors = {k: [] for k in ATTRIBUTES}

    for img_name in tqdm(os.listdir(image_dir)):
        img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        emb = encode_image(model, img_tensor, cfg["device"])[0]

        for attr in vectors:
            vectors[attr].append(emb)

        image_ids.append(img_name)

    os.makedirs("data/indexes", exist_ok=True)

    for attr in vectors:
        index = build_index(np.array(vectors[attr]))
        faiss.write_index(index, f"data/indexes/{attr}.faiss")

    json.dump(image_ids, open("data/indexes/image_ids.json", "w"))

if __name__ == "__main__":
    main()
