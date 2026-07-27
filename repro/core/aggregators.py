import torch
import numpy as np

def robust_aggregator(vectors, agg_type='avg', **kwargs):
    if agg_type == 'avg':
        return torch.stack(vectors).mean(dim=0)
    elif agg_type == 'krum':
        return krum(vectors, **kwargs)
    elif agg_type == 'cm':
        return coordinate_wise_median(vectors)
    elif agg_type == 'cclip':
        return centered_clipping(vectors, **kwargs)
    elif agg_type == 'rfa':
        return rfa_aggregator(vectors, **kwargs)
    else:
        raise ValueError(f"Unknown aggregator: {agg_type}")

def coordinate_wise_median(vectors):
    stacked = torch.stack(vectors)
    return stacked.median(dim=0)[0]

def krum(vectors, f=1):
    n = len(vectors)
    stacked = torch.stack(vectors)
    distances = torch.cdist(stacked, stacked, p=2)
    scores = distances.sum(dim=1)
    if f >= n:
        f = n - 1
    _, indices = torch.topk(scores, n - f, largest=False)
    return stacked[indices].mean(dim=0)

def centered_clipping(vectors, tau=10.0, max_iter=5):
    stacked = torch.stack(vectors)
    center = stacked.mean(dim=0)
    for _ in range(max_iter):
        diffs = stacked - center
        norms = torch.norm(diffs, dim=1, keepdim=True)
        scales = torch.where(norms > tau, tau / (norms + 1e-8),
                             torch.ones_like(norms))
        clipped = center + diffs * scales
        center = clipped.mean(dim=0)
    return center

def rfa_aggregator(vectors, max_iter=20, tol=1e-6):
    stacked = torch.stack(vectors)
    geo_median = geometric_median(stacked, max_iter, tol)
    return geo_median

def geometric_median(points, max_iter=20, tol=1e-6):
    estimate = points.mean(dim=0)
    for _ in range(max_iter):
        diffs = points - estimate
        norms = torch.norm(diffs, dim=1, keepdim=True)
        weights = 1.0 / (norms + 1e-8)
        weights = weights / weights.sum()
        new_estimate = (weights * points).sum(dim=0)
        if torch.norm(new_estimate - estimate) < tol:
            return new_estimate
        estimate = new_estimate
    return estimate
