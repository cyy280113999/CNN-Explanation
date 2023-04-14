import torch
from itertools import product

def binarize(mask, sparsity=0.5):
    x = mask.flatten()
    n = len(x)
    ascending = x.sort()[0]
    value = ascending[int(sparsity * n)]
    return 1.0 * (mask >= value)


def positize(mask):
    return 1.0 * (mask >= 0)


def binarize_mul_noisy_n(mask, sparsity=0.5, top=True, std=0.1):
    x = mask.flatten()
    threshold = x.sort()[0][int(sparsity * len(x))]
    if top:
        mask = 1.0 * (mask <= threshold) + std * torch.randn_like(mask) * (mask > threshold)
    else:
        mask = 1.0 * (mask >= threshold) + std * torch.randn_like(mask) * (mask < threshold)
    return mask


def binarize_add_noisy_n(mask, sparsity=0.5, top=True, std=0.1):
    x = mask.flatten()
    threshold = x.sort()[0][int(sparsity * len(x))]
    if top:
        mask = std * torch.randn_like(mask) * (mask > threshold)
    else:
        mask = std * torch.randn_like(mask) * (mask < threshold)
    return mask

def maximalLoc(heatmap2d, top=True):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    assert len(heatmap2d.shape) == 2
    h, w = heatmap2d.shape
    f = heatmap2d.flatten()
    loc = f.sort(descending=top)[1][0].item()
    x = loc // h
    y = loc - (x * h)
    return x,y

def patch(heatmap2d, loc, r=1):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    h,w=heatmap2d.shape
    x,y=loc
    xL = max(0, x - r)
    xH = min(h, x + r)
    yL = max(0, y - r)
    yH = min(w, y + r)
    patched = torch.ones_like(heatmap2d)
    patched[xL:xH + 1, yL:yH + 1] = 0
    return patched

def maximalPatch(heatmap2d, top=True, r=1):
    if len(heatmap2d.shape) == 4:
        heatmap2d = heatmap2d.squeeze(0)
    if len(heatmap2d.shape) == 3:
        heatmap2d = heatmap2d.sum(0)
    assert len(heatmap2d.shape) == 2
    h, w = heatmap2d.shape
    f = heatmap2d.flatten()
    loc = f.sort(descending=top)[1][0].item()
    x = loc // h
    y = loc - (x * h)

    xL = max(0, x - r)
    xH = min(h, x + r)
    yL = max(0, y - r)
    yH = min(w, y + r)
    patched = torch.ones_like(heatmap2d)
    patched[xL:xH + 1, yL:yH + 1] = 0
    return patched



def cornerMask(mask2d, r=1):
    if len(mask2d.shape) == 4:
        mask2d = mask2d.squeeze(0)
    if len(mask2d.shape) == 3:
        mask2d = mask2d.sum(0)
    assert len(mask2d.shape) == 2
    h, w = mask2d.shape
    patched = torch.ones_like(mask2d)
    for x, y in product((0,h-1),(0,w-1)):
        xL = max(0, x - r)
        xH = min(h, x + r)
        yL = max(0, y - r)
        yH = min(w, y + r)
        patched[xL:xH + 1, yL:yH + 1] = 0
    return patched