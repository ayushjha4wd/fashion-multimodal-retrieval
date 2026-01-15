import torch

def encode_text(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()
