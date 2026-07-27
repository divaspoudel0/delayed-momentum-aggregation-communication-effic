#!/usr/bin/env python3
"""Reproduce Claim 1: D-Byz-SGDM maintains convergence under partial participation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_claim1_experiment
import json

def main():
    device = 'cpu'
    print(f"Running on device: {device}")
    results = run_claim1_experiment(device)

    summary = {}
    for attack, algo_results in results.items():
        summary[attack] = {
            name: {'final_accuracy': r['final_accuracy'],
                   'time_seconds': r['time_seconds']}
            for name, r in algo_results.items()
        }

    print("\n" + "=" * 60)
    print("Summary of Claim 1 Results")
    print("=" * 60)
    print(f"{'Attack':>8} | {'FedAvg':>7} | {'FedAvg-M':>8} | {'D-Byz-SGDM':>10} | {'Byz-VR':>7}")
    print("-" * 60)
    for attack, algo_results in sorted(summary.items()):
        fa = algo_results.get('FedAvg', {}).get('final_accuracy', 0)
        fam = algo_results.get('FedAvg-M', {}).get('final_accuracy', 0)
        dbs = algo_results.get('D-Byz-SGDM', {}).get('final_accuracy', 0)
        bvm = algo_results.get('Byz-VR-MARINA-PP', {}).get('final_accuracy', 0)
        print(f"{attack:>8} | {fa:>6.1f}% | {fam:>7.1f}% | {dbs:>9.1f}% | {bvm:>6.1f}%")

    with open('results_claim1.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to results_claim1.json")

if __name__ == '__main__':
    main()
