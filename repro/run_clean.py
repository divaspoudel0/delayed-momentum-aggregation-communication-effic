#!/usr/bin/env python3
"""Clean synthetic experiment to verify key claims."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

torch.set_num_threads(1)

def run_comparison(n_clients=10, n_byz=3, p=0.5, n_steps=200, attack='ipm',
                   agg_type='avg', lr=0.05, momentum=0.9, dim=10):
    rng = np.random.RandomState(42)
    true_w = torch.FloatTensor(rng.randn(dim) / np.sqrt(dim))
    
    client_X = [torch.FloatTensor(rng.randn(50, dim)) for _ in range(n_clients)]
    client_y = [cx @ true_w + 0.05 * torch.randn(50) for cx in client_X]
    test_X = torch.FloatTensor(rng.randn(200, dim))
    test_y = test_X @ true_w + 0.05 * torch.randn(200)
    
    byz_ids = set(range(n_clients - n_byz, n_clients))
    
    def mse(model_w, X, y):
        return ((X @ model_w - y) ** 2).mean().item()
    
    results = {}
    
    # D-Byz-SGDM
    w = torch.zeros(dim)
    client_mom = [torch.zeros(dim) for _ in range(n_clients)]
    dbyz_errs = []
    for t in range(n_steps):
        sampled = set(np.where(rng.random(n_clients) < p)[0])
        momentums = []
        for i in range(n_clients):
            if i in sampled:
                g = 2 * client_X[i].t() @ (client_X[i] @ w - client_y[i]) / len(client_X[i])
                mi = (1 - momentum) * client_mom[i] + momentum * g
                client_mom[i] = mi
            else:
                mi = client_mom[i]
            if i in byz_ids:
                if attack == 'bf':
                    mi = -mi
                elif attack == 'ipm':
                    mi = -2.0 * mi
                elif attack == 'inf':
                    mi = 100 * torch.ones_like(mi) * torch.norm(mi)
            momentums.append(mi)
        stacked = torch.stack(momentums)
        if agg_type == 'avg':
            agg = stacked.mean(dim=0)
        elif agg_type == 'cclip':
            c = stacked.mean(dim=0)
            d = stacked - c
            nrm = torch.norm(d, dim=1, keepdim=True)
            s = torch.where(nrm > 5.0, 5.0 / (nrm + 1e-8), torch.ones_like(nrm))
            agg = (c + d * s).mean(dim=0)
        elif agg_type == 'cm':
            agg = stacked.median(dim=0)[0]
        else:
            agg = stacked.mean(dim=0)
        w -= lr * agg
        dbyz_errs.append(mse(w, test_X, test_y))
    
    # FedAvg (aggregates only sampled clients)
    w = torch.zeros(dim)
    fa_errs = []
    for t in range(n_steps):
        sampled = set(np.where(rng.random(n_clients) < p)[0])
        grads = []
        for i in sampled:
            g = 2 * client_X[i].t() @ (client_X[i] @ w - client_y[i]) / len(client_X[i])
            if i in byz_ids:
                if attack == 'bf':
                    g = -g
                elif attack == 'ipm':
                    g = -2.0 * g
                elif attack == 'inf':
                    g = 100 * torch.ones_like(g) * torch.norm(g)
            grads.append(g)
        if not grads:
            continue
        stacked = torch.stack(grads)
        if agg_type == 'avg':
            agg = stacked.mean(dim=0)
        elif agg_type == 'cclip':
            c = stacked.mean(dim=0)
            d = stacked - c
            nrm = torch.norm(d, dim=1, keepdim=True)
            s = torch.where(nrm > 5.0, 5.0 / (nrm + 1e-8), torch.ones_like(nrm))
            agg = (c + d * s).mean(dim=0)
        else:
            agg = stacked.mean(dim=0)
        w -= lr * agg
        fa_errs.append(mse(w, test_X, test_y))
    
    # FedAvg-M (server momentum, only sampled clients)
    w = torch.zeros(dim)
    sv_mom = torch.zeros(dim)
    fam_errs = []
    for t in range(n_steps):
        sampled = set(np.where(rng.random(n_clients) < p)[0])
        grads = []
        for i in sampled:
            g = 2 * client_X[i].t() @ (client_X[i] @ w - client_y[i]) / len(client_X[i])
            if i in byz_ids:
                if attack == 'bf':
                    g = -g
                elif attack == 'ipm':
                    g = -2.0 * g
                elif attack == 'inf':
                    g = 100 * torch.ones_like(g) * torch.norm(g)
            grads.append(g)
        if not grads:
            continue
        stacked = torch.stack(grads)
        if agg_type == 'avg':
            agg = stacked.mean(dim=0)
        elif agg_type == 'cclip':
            c = stacked.mean(dim=0)
            d = stacked - c
            nrm = torch.norm(d, dim=1, keepdim=True)
            s = torch.where(nrm > 5.0, 5.0 / (nrm + 1e-8), torch.ones_like(nrm))
            agg = (c + d * s).mean(dim=0)
        else:
            agg = stacked.mean(dim=0)
        sv_mom = momentum * sv_mom + (1 - momentum) * agg
        w -= lr * sv_mom
        fam_errs.append(mse(w, test_X, test_y))
    
    return {
        'D-Byz-SGDM': dbyz_errs[-1],
        'FedAvg': fa_errs[-1],
        'FedAvg-M': fam_errs[-1],
        'D-Byz-SGDM_curve': dbyz_errs,
        'FedAvg_curve': fa_errs,
        'FedAvg-M_curve': fam_errs,
    }

def main():
    print("=" * 70)
    print("D-Byz-SGDM vs FedAvg vs FedAvg-M — Synthetic Linear Regression")
    print("=" * 70)
    
    configs = [
        # (desc, n, n_byz, p, attack, agg)
        ("Clean (no attack)", 10, 0, 0.5, 'none', 'avg'),
        ("BF attack, avg agg", 10, 3, 0.5, 'bf', 'avg'),
        ("IPM attack, avg agg", 10, 3, 0.5, 'ipm', 'avg'),
        ("IPM, cclip agg", 10, 3, 0.5, 'ipm', 'cclip'),
        ("IPM, p=0.3 (high Byz risk)", 10, 3, 0.3, 'ipm', 'cclip'),
        ("Extreme: 40% Byz, p=0.3", 10, 4, 0.3, 'ipm', 'cclip'),
    ]
    
    all_results = {}
    for desc, n, n_byz, p, attack, agg in configs:
        print(f"\n  {desc}")
        r = run_comparison(n_clients=n, n_byz=n_byz, p=p, attack=attack, agg_type=agg)
        all_results[desc] = r
        for method in ['D-Byz-SGDM', 'FedAvg', 'FedAvg-M']:
            print(f"    {method:>12}: final MSE = {r[method]:.6f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("D-Byz-SGDM: Aggregates over ALL n clients (sampled + cached)")
    print("  → Byzantine fraction in aggregation = δ < 0.5 ALWAYS")
    print("FedAvg/FedAvg-M: Aggregates over sampled clients only")
    print("  → Byzantine can dominate sampled set (especially at low p)")
    
    os.makedirs('results', exist_ok=True)
    with open('results/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to results/comparison_results.json")

if __name__ == '__main__':
    main()
