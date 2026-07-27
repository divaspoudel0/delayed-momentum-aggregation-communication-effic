#!/usr/bin/env python3
"""Better-tuned synthetic experiment."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

def run(n_clients=20, delta=0.2, p=0.5, n_steps=100, lr=0.05, alpha=0.9):
    n_byz = int(n_clients * delta)
    byz_ids = set(range(n_clients - n_byz, n_clients))
    dim = 20
    
    true_w = torch.randn(dim) / np.sqrt(dim)
    data = [(torch.randn(20, dim), (torch.randn(20, dim) @ true_w + 0.01 * torch.randn(20))) 
            for _ in range(n_clients)]
    test_X, test_y = torch.randn(500, dim), torch.randn(500, dim) @ true_w + 0.01 * torch.randn(500)
    
    def test_err(w):
        return ((test_X @ w - test_y) ** 2).mean().item()
    
    def gradient(w, X, y):
        return 2 * X.t() @ (X @ w - y) / len(y)
    
    results = {}
    
    # D-Byz-SGDM with cclip
    for agg in ['cclip', 'avg']:
        for attack in ['none', 'bf', 'ipm']:
            w = torch.zeros(dim)
            cm = [torch.zeros(dim) for _ in range(n_clients)]
            errs = []
            for t in range(n_steps):
                sampled = set(np.where(np.random.random(n_clients) < p)[0])
                moms = []
                for i in range(n_clients):
                    if i in sampled:
                        g = gradient(w, *data[i])
                        mi = (1 - alpha) * cm[i] + alpha * g
                        cm[i] = mi
                    else:
                        mi = cm[i]
                    if i in byz_ids and attack != 'none':
                        if attack == 'bf':
                            mi = -mi
                        elif attack == 'ipm':
                            mi = -0.5 * mi
                    moms.append(mi)
                stacked = torch.stack(moms)
                if agg == 'cclip':
                    center = stacked.mean(dim=0)
                    diffs = stacked - center
                    norms = torch.norm(diffs, dim=1, keepdim=True)
                    scales = torch.where(norms > 2.0, 2.0 / (norms + 1e-8), torch.ones_like(norms))
                    agg_grad = (center + diffs * scales).mean(dim=0)
                else:
                    agg_grad = stacked.mean(dim=0)
                w -= lr * agg_grad
                errs.append(test_err(w))
            results[f'DByz_{agg}_{attack}'] = errs[-1]
            print(f"  D-Byz-SGDM ({agg:>5}, {attack:>4}): {errs[-1]:.6f}")
        
        # FedAvg (no momentum, sampled only)
        for attack in ['none', 'bf', 'ipm']:  
            w = torch.zeros(dim)
            errs = []
            for t in range(n_steps):
                sampled = set(np.where(np.random.random(n_clients) < p)[0])
                grads = []
                for i in sampled:
                    g = gradient(w, *data[i])
                    if i in byz_ids and attack != 'none':
                        if attack == 'bf':
                            g = -g
                        elif attack == 'ipm':
                            g = -0.5 * g
                    grads.append(g)
                if not grads:
                    continue
                stacked = torch.stack(grads)
                if agg == 'cclip':
                    center = stacked.mean(dim=0)
                    diffs = stacked - center
                    norms = torch.norm(diffs, dim=1, keepdim=True)
                    scales = torch.where(norms > 2.0, 2.0 / (norms + 1e-8), torch.ones_like(norms))
                    agg_grad = (center + diffs * scales).mean(dim=0)
                else:
                    agg_grad = stacked.mean(dim=0)
                w -= lr * agg_grad
                errs.append(test_err(w))
            results[f'FedAvg_{agg}_{attack}'] = errs[-1]
            print(f"  FedAvg    ({agg:>5}, {attack:>4}): {errs[-1]:.6f}")
    
    # Clean baseline (no Byzantine, full participation)
    w = torch.zeros(dim)
    for t in range(n_steps):
        g = sum(gradient(w, *data[i]) for i in range(n_clients)) / n_clients
        w -= lr * g
    clean_err = test_err(w)
    results['clean_baseline'] = clean_err
    print(f"  Clean baseline            : {clean_err:.6f}")
    
    return results

results = run()
print("\n" + "=" * 60)
print("Summary: Lower MSE = Better Convergence")
print("=" * 60)
with open('results/tuned_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
