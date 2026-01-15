import yaml
import numpy as np
from models.text_encoder import encode_text
from models.attribute_heads import build_prompts

def extract_attribute_embedding(model, tokenizer, attr_name, values, device):
    prompts = yaml.safe_load(open("config/prompts.yaml"))[attr_name]
    text_prompts = build_prompts(prompts, values)
    embeddings = encode_text(model, tokenizer, text_prompts, device)
    return embeddings.mean(axis=0)
