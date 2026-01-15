import torch

def encode_image(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()
