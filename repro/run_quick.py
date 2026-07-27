#!/usr/bin/env python3
"""Quick reproduction experiments with MLP model."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.models import MLP_MNIST
from core.algorithms import DByzSGDM, FedAvg, FedAvgM
from core.utils import prepare_mnist_data
from experiments.runner import run_experiment

N_CLIENTS = 10
N_EPOCHS = 5
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
P = 0.5
DELTA = 0.2

def run_all():
    results = {}

    for iid_flag, iid_name in [(True, 'iid'), (False, 'noniid')]:
        print(f"\n{'='*50}")
        print(f"  {iid_name.upper()} Setting")
        print(f"{'='*50}")
        client_loaders, test_loader = prepare_mnist_data(
            N_CLIENTS, BATCH_SIZE, iid=iid_flag, seed=42)

        # Claim 1: Byzantine robustness with cclip
        print(f"\n  --- Byzantine Attacks (cclip aggregator) ---")
        for attack in ['bf', 'ipm', 'alie']:
            results[f"{iid_name}_byz_{attack}"] = {}
            for name, Cls, kwargs in [
                ('FedAvg', FedAvg, {'lr': LR, 'agg_type': 'cclip'}),
                ('FedAvg-M', FedAvgM, {'lr': LR, 'momentum': MOMENTUM, 'agg_type': 'cclip'}),
                ('D-Byz-SGDM', DByzSGDM, {'lr': LR, 'momentum': MOMENTUM, 'agg_type': 'cclip'}),
            ]:
                opt = Cls(MLP_MNIST, N_CLIENTS, DELTA, P, seed=42, **kwargs)
                t0 = time.time()
                res = run_experiment(opt, MLP_MNIST, client_loaders, test_loader,
                                     N_EPOCHS, attack, 'cpu', seed=42)
                results[f"{iid_name}_byz_{attack}"][name] = {
                    'final_acc': res['final_accuracy'],
                    'acc_curve': res['accuracy'],
                    'time': time.time() - t0
                }
                print(f"  {attack:>5} | {name:>12}: {res['final_accuracy']:.1f}%")

        # Claim 5: Non-Byzantine (avg aggregator)
        print(f"\n  --- No Byzantine (avg aggregator) ---")
        results[f"{iid_name}_clean"] = {}
        for name, Cls, kwargs in [
            ('FedAvg', FedAvg, {'lr': LR, 'agg_type': 'avg'}),
            ('FedAvg-M', FedAvgM, {'lr': LR, 'momentum': MOMENTUM, 'agg_type': 'avg'}),
            ('D-Byz-SGDM', DByzSGDM, {'lr': LR, 'momentum': MOMENTUM, 'agg_type': 'avg'}),
        ]:
            opt = Cls(MLP_MNIST, N_CLIENTS, 0.0, P, seed=42, **kwargs)
            t0 = time.time()
            res = run_experiment(opt, MLP_MNIST, client_loaders, test_loader,
                                 N_EPOCHS, 'none', 'cpu', seed=42)
            results[f"{iid_name}_clean"][name] = {
                'final_acc': res['final_accuracy'],
                'acc_curve': res['accuracy'],
                'time': time.time() - t0
            }
            print(f"  {'none':>5} | {name:>12}: {res['final_accuracy']:.1f}%")

    print(f"\n{'='*50}")
    print("  SUMMARY")
    print(f"{'='*50}")
    for key, algos in results.items():
        print(f"\n  {key}:")
        for name, r in algos.items():
            print(f"    {name:>12}: {r['final_acc']:.1f}% ({r['time']:.1f}s)")

    with open('results_quick.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results_quick.json")

if __name__ == '__main__':
    run_all()
