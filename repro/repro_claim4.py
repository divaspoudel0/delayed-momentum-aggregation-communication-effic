#!/usr/bin/env python3
"""Reproduce Claim 4: MNIST CNN experiments with 6 attacks x 5 aggregators."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_claim4_experiment
import json

def main():
    device = 'cpu'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduced', action='store_true',
                        help='Run reduced version (fewer attacks/aggregators)')
    args = parser.parse_args()

    print(f"Running on device: {device}")
    results = run_claim4_experiment(device, reduced=args.reduced)

    summary = {}
    for key, algo_results in results.items():
        summary[key] = {
            name: r['final_accuracy']
            for name, r in algo_results.items()
        }

    print("\n" + "=" * 80)
    print("Claim 4 Summary (Final Accuracies %)")
    print("=" * 80)
    header = f"{'Config':>20}"
    algos = ['FedAvg', 'FedAvg-M', 'D-Byz-SGDM', 'Byz-VR-MARINA-PP']
    for a in algos:
        header += f" | {a:>15}"
    print(header)
    print("-" * 80)
    for key, accs in sorted(summary.items()):
        line = f"{key:>20}"
        for a in algos:
            v = accs.get(a, 0)
            line += f" | {v:>14.1f}%"
        print(line)

    with open('results_claim4.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nResults saved to results_claim4.json")

if __name__ == '__main__':
    main()
