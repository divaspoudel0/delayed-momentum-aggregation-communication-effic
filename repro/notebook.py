# ---
# jupyter:
#   marimo:
#     name: "repro-delayed-momentum-aggregation"
# ---

import sys, os, json, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))
from core.models import CNN_MNIST, MLP_MNIST
from core.algorithms import DByzSGDM, FedAvg, FedAvgM, ByzVRMARINAPP
from core.utils import prepare_mnist_data, prepare_cifar10_data, evaluate
from experiments.runner import run_experiment

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Quick Verification ===
print("=" * 60)
print("Quick verification of D-Byz-SGDM vs FedAvg under IPM attack")
print("=" * 60)

n_clients = 10
n_epochs = 5
client_loaders, test_loader = prepare_mnist_data(n_clients, 128, iid=True, seed=42)

results = {}
for name, Cls, kwargs in [
    ("FedAvg", FedAvg, {"lr": 0.01, "agg_type": "cclip"}),
    ("FedAvg-M", FedAvgM, {"lr": 0.01, "momentum": 0.9, "agg_type": "cclip"}),
    ("D-Byz-SGDM", DByzSGDM, {"lr": 0.01, "momentum": 0.9, "agg_type": "cclip"}),
]:
    opt = Cls(CNN_MNIST, n_clients, 0.2, 0.5, seed=42, **kwargs)
    res = run_experiment(opt, CNN_MNIST, client_loaders, test_loader,
                         n_epochs, "ipm", device, seed=42)
    results[name] = res["final_accuracy"]
    print(f"  {name:>12}: {res['final_accuracy']:.2f}%")

print(f"\nD-Byz-SGDM > FedAvg/FedAvg-M under IPM: {results.get('D-Byz-SGDM', 0) > results.get('FedAvg', 0)}")

# === Full Grid (GPU) ===
print("\n" + "=" * 60)
print("Full: 6 attacks × 5 aggregators × 4 algorithms (GPU recommended)")
print("=" * 60)

attacks = ["bf", "lf", "mimic", "ipm", "alie", "inf"]
aggregators = ["avg", "krum", "cm", "cclip", "rfa"]
algos = [
    ("FedAvg", FedAvg, {"lr": 0.01}),
    ("FedAvg-M", FedAvgM, {"lr": 0.01, "momentum": 0.9}),
    ("D-Byz-SGDM", DByzSGDM, {"lr": 0.01, "momentum": 0.9}),
    ("Byz-VR-MARINA-PP", ByzVRMARINAPP, {"lr": 0.01, "clip_lambda": 10.0}),
]

all_results = {}
grid_size = len(attacks) * len(aggregators) * len(algos)
print(f"Total configurations: {grid_size}")

for agg in aggregators:
    for attack in attacks:
        key = f"{agg}_{attack}"
        all_results[key] = {}
        for name, Cls, base_kwargs in algos:
            kwargs = {**base_kwargs, "agg_type": agg}
            opt = Cls(CNN_MNIST, n_clients, 0.2, 0.5, seed=42, **kwargs)
            res = run_experiment(opt, CNN_MNIST, client_loaders, test_loader,
                                 10, attack, device, seed=42)
            all_results[key][name] = res["final_accuracy"]

with open("results/grid_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nGrid results saved to results/grid_results.json")
