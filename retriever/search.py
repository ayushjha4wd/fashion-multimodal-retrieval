import yaml, faiss, json
import numpy as np

from models.text_encoder import encode_text
from models.clip_loader import load_clip
from retriever.parse_query import parse_query
from retriever.rank_fusion import fuse

def main():
    cfg = yaml.safe_load(open("config/model_config.yaml"))
    weights = yaml.safe_load(open("config/weights.yaml"))

    model, _, tokenizer = load_clip(
        cfg["clip_model"], cfg["pretrained"], cfg["device"]
    )

    query = input("Enter query: ")
    attrs = parse_query(query)

    image_ids = json.load(open("data/indexes/image_ids.json"))
    scores_per_attr = {}

    for attr in attrs:
        index = faiss.read_index(f"data/indexes/{attr}.faiss")
        emb = encode_text(model, tokenizer, [query], cfg["device"])
        scores, _ = index.search(emb.astype(np.float32), 1)
        scores_per_attr[attr] = scores[0][0]

    final_score = fuse(scores_per_attr, weights)
    print("Final similarity score:", final_score)

if __name__ == "__main__":
    main()
