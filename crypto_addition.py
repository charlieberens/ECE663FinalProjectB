import torch

def create_mask(
    H, W, args
):
    # This embeds the message in every nth pixel
    n = int(args)

    mask = torch.zeros(3,H,W)
    
    for i in range(H):
        for j in range(W):
            mask[:,i,j] = int((i*W + j) % n == 0)

    return mask, 1 - mask
