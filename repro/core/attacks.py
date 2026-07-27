import torch
import numpy as np

def byzantine_attack(gradients, attack_type='none', **kwargs):
    if attack_type == 'none':
        return gradients
    elif attack_type == 'bf':
        return bit_flip_attack(gradients)
    elif attack_type == 'lf':
        return label_flip_attack(gradients, **kwargs)
    elif attack_type == 'mimic':
        return mimic_attack(gradients, **kwargs)
    elif attack_type == 'ipm':
        return ipm_attack(gradients, **kwargs)
    elif attack_type == 'alie':
        return alie_attack(gradients, **kwargs)
    elif attack_type == 'inf':
        return inf_attack(gradients, **kwargs)
    else:
        raise ValueError(f"Unknown attack: {attack_type}")

def bit_flip_attack(gradients):
    return [-g for g in gradients]

def label_flip_attack(gradients, delta=0.1):
    return [g + delta * torch.randn_like(g) * torch.norm(g) for g in gradients]

def mimic_attack(gradients, noise_scale=0.1):
    if len(gradients) < 2:
        return gradients
    honest_grads = [g for i, g in enumerate(gradients) if i < len(gradients) // 2]
    if not honest_grads:
        return gradients
    avg_honest = torch.stack(honest_grads).mean(dim=0)
    noise = noise_scale * torch.randn_like(avg_honest) * torch.norm(avg_honest)
    return [avg_honest + noise for _ in gradients]

def ipm_attack(gradients, scale=None):
    if len(gradients) < 2:
        return gradients
    honest_grads = [g for i, g in enumerate(gradients) if i < len(gradients) // 2]
    if not honest_grads:
        return gradients
    avg_honest = torch.stack(honest_grads).mean(dim=0)
    if scale is None:
        scale = 1.0
    return [-scale * avg_honest for _ in gradients]

def alie_attack(gradients, epsilon=1.0):
    if len(gradients) < 2:
        return gradients
    honest_grads = [g for i, g in enumerate(gradients) if i < len(gradients) // 2]
    if not honest_grads:
        return gradients
    stacked = torch.stack(honest_grads)
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0)
    z = epsilon * sigma
    noisy_updates = mu - z
    return [noisy_updates for _ in gradients]

def inf_attack(gradients, scale=100.0):
    if not gradients:
        return gradients
    ref = gradients[0]
    inf_val = scale * torch.ones_like(ref) * torch.norm(ref)
    return [inf_val for _ in gradients]
