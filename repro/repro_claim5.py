#!/usr/bin/env python3
"""Reproduce Claim 5: Non-Byzantine setting comparison."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_claim5_experiment
import json

def main():
    device = 'cpu'
    print(f"Running on device: {device}")
    results = run_claim5_experiment(device)

    print("\n" + "=" * 70)
    print("Claim 5 Summary: Non-Byzantine Setting (δ=0)")
    print("=" * 70)
    for setting, algo_results in results.items():
        print(f"\n  {setting.upper()} Setting:")
        for name, r in algo_results.items():
            print(f"    {name:>20}: Final Acc = {r['final_accuracy']:.2f}%")

    with open('results_claim5.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to results_claim5.json")

if __name__ == '__main__':
    main()
