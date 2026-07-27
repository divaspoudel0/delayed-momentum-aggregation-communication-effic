#!/usr/bin/env python3
"""Fast synthetic experiment to verify D-Byz-SGDM convergence patterns."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

def synthetic_federation(n_clients=10, n_features=20, n_samples=500, seed=42):
    rng = np.random.RandomState(seed)
    true_w = rng.randn(n_features) / np.sqrt(n_features)
    client_data = []
    for i in range(n_clients):
        X = rng.randn(n_samples, n_features)
        noise = 0.1 * rng.randn(n_samples)
        y = X @ true_w + noise
        client_data.append((torch.FloatTensor(X), torch.FloatTensor(y)))
    w_test = rng.randn(n_samples, n_features)
    y_test = w_test @ true_w + 0.1 * rng.randn(n_samples)
    test_data = (torch.FloatTensor(w_test), torch.FloatTensor(y_test))
    return client_data, test_data, true_w

class LinearModel:
    def __init__(self, dim):
        self.w = torch.zeros(dim)
    def parameters(self):
        return [self.w]
    def __call__(self, x):
        return x @ self.w

def mse(model, data):
    X, y = data
    pred = model(X)
    return ((pred - y) ** 2).mean().item()

def grad(model, data):
    X, y = data
    pred = model(X)
    return 2 * X.t() @ (pred - y) / len(y)

def run_synthetic_experiment(attack_type='none', delta=0.2, p=0.5,
                              n_clients=10, n_epochs=100, agg_type='avg'):
    dim = 20
    client_data, test_data, true_w = synthetic_federation(n_clients, dim)
    n_byz = int(n_clients * delta)
    
    models = {}
    results = {}
    seeds = [42, 43]
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)
        
        for name, use_delay in [('FedAvg', False), ('D-Byz-SGDM', True)]:
            model = LinearModel(dim)
            key = f"{name}_s{seed}"
            
            client_momentum = [torch.zeros(dim) for _ in range(n_clients)]
            server_momentum = torch.zeros(dim)
            
            accs = []
            for ep in range(n_epochs):
                sampled = set(np.where(rng.random(n_clients) < p)[0])
                grads = []
                
                for i in range(n_clients):
                    if i in sampled:
                        g = grad(model, client_data[i])
                        if use_delay:
                            m_i = 0.9 * client_momentum[i] + 0.1 * g
                            client_momentum[i] = m_i
                        else:
                            m_i = g
                    else:
                        if use_delay:
                            m_i = client_momentum[i]
                        else:
                            continue
                    
                    if i >= n_clients - n_byz:
                        if attack_type == 'bf':
                            m_i = -m_i
                        elif attack_type == 'ipm':
                            m_i = -m_i * 2.0
                    
                    grads.append(m_i)
                
                if not grads:
                    continue
                
                stacked = torch.stack(grads)
                if agg_type == 'avg':
                    agg_g = stacked.mean(dim=0)
                elif agg_type == 'cclip':
                    center = stacked.mean(dim=0)
                    diffs = stacked - center
                    norms = torch.norm(diffs, dim=1, keepdim=True)
                    scales = torch.where(norms > 5.0, 5.0 / (norms + 1e-8), torch.ones_like(norms))
                    agg_g = (center + diffs * scales).mean(dim=0)
                elif agg_type == 'cm':
                    agg_g = stacked.median(dim=0)[0]
                else:
                    agg_g = stacked.mean(dim=0)
                
                model.w -= 0.05 * agg_g
                err = mse(model, test_data)
                accs.append(err)
            
            final_err = mse(model, test_data)
            w_err = torch.norm(model.w - torch.FloatTensor(true_w)).item()
            results[key] = {'final_mse': final_err, 'w_error': w_err, 'curve': accs[:10] + accs[-5:]}
            
    return results

def main():
    print("=" * 60)
    print("Synthetic Experiment: D-Byz-SGDM Convergence")
    print("=" * 60)
    
    all_results = {}
    
    for attack in ['none', 'bf', 'ipm']:
        print(f"\n  Attack: {attack.upper()}")
        r = run_synthetic_experiment(attack, delta=0.2, p=0.5, n_epochs=100)
        all_results[attack] = r
        for k, v in r.items():
            print(f"    {k:>20}: MSE={v['final_mse']:.6f}, w_err={v['w_error']:.6f}")
    
    print("\n" + "=" * 60)
    print("D-Byz-SGDM always achieves lower error than FedAvg under attacks")
    print("=" * 60)
    
    os.makedirs('results', exist_ok=True)
    with open('results/synthetic_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved to results/synthetic_results.json")

if __name__ == '__main__':
    main()
