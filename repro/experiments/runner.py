import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
import numpy as np
import json
import time
from tqdm import tqdm
from core.utils import evaluate, compute_loss

def run_experiment(optimizer, model_fn, client_loaders, test_loader,
                   n_epochs=10, attack_type='none', device='cpu',
                   log_interval=1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    global_model = model_fn()
    results = {
        'epoch': [],
        'accuracy': [],
        'loss': [],
        'client_counts': []
    }

    if hasattr(optimizer, 'init_state'):
        optimizer.init_state(global_model)

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        sampled = optimizer.step(global_model, client_loaders, attack_type)

        if epoch % log_interval == 0 or epoch == 1:
            acc = evaluate(global_model, test_loader)
            loss = compute_loss(global_model, test_loader)
            results['epoch'].append(epoch)
            results['accuracy'].append(acc)
            results['loss'].append(loss)
            results['client_counts'].append(len(sampled))
            print(f"  Epoch {epoch:3d}: Acc={acc:.2f}%, Loss={loss:.4f}, Clients={len(sampled)}")

    elapsed = time.time() - start_time
    results['final_accuracy'] = results['accuracy'][-1] if results['accuracy'] else 0.0
    results['time_seconds'] = elapsed
    results['n_epochs'] = n_epochs

    return results


def run_claim1_experiment(device='cpu'):
    from core.models import CNN_MNIST
    from core.algorithms import DByzSGDM, FedAvg, FedAvgM, ByzVRMARINAPP
    from core.utils import prepare_mnist_data

    print("=" * 60)
    print("Claim 1: D-Byz-SGDM Convergence under Partial Participation")
    print("=" * 60)

    configs = {
        'n_clients': 25,
        'byzantine_ratio': 0.2,
        'participation_prob': 0.5,
        'n_epochs': 10,
        'batch_size': 64,
        'lr': 0.01,
        'momentum': 0.9,
        'agg_type': 'cclip'
    }

    client_loaders_iid, test_loader = prepare_mnist_data(
        configs['n_clients'], configs['batch_size'], iid=True, seed=42)

    attacks = ['bf', 'lf', 'mimic', 'ipm', 'alie', 'inf']
    all_results = {}

    for attack in attacks:
        print(f"\n--- Attack: {attack.upper()} ---")

        optimizers = {
            'FedAvg': FedAvg(
                CNN_MNIST, configs['n_clients'], configs['byzantine_ratio'],
                configs['participation_prob'], configs['lr'],
                configs['agg_type'], seed=42),
            'FedAvg-M': FedAvgM(
                CNN_MNIST, configs['n_clients'], configs['byzantine_ratio'],
                configs['participation_prob'], configs['lr'],
                configs['momentum'], configs['agg_type'], seed=42),
            'D-Byz-SGDM': DByzSGDM(
                CNN_MNIST, configs['n_clients'], configs['byzantine_ratio'],
                configs['participation_prob'], configs['lr'],
                configs['momentum'], configs['agg_type'], seed=42),
            'Byz-VR-MARINA-PP': ByzVRMARINAPP(
                CNN_MNIST, configs['n_clients'], configs['byzantine_ratio'],
                configs['participation_prob'], configs['lr'],
                clip_lambda=10.0, seed=42),
        }

        attack_results = {}
        for name, optimizer in optimizers.items():
            print(f"\n  Optimizer: {name}")
            torch.manual_seed(42)
            res = run_experiment(
                optimizer, CNN_MNIST, client_loaders_iid, test_loader,
                configs['n_epochs'], attack, device, seed=42)
            attack_results[name] = res
            print(f"  -> Final Acc: {res['final_accuracy']:.2f}%")

        all_results[attack] = attack_results

    return all_results


def run_claim4_experiment(device='cpu', reduced=False):
    from core.models import CNN_MNIST
    from core.algorithms import DByzSGDM, FedAvg, FedAvgM, ByzVRMARINAPP
    from core.utils import prepare_mnist_data

    print("=" * 60)
    print("Claim 4: MNIST CNN Experiments")
    print("=" * 60)

    if reduced:
        attacks = ['bf', 'ipm', 'alie']
        aggregators = ['avg', 'cclip', 'cm']
        n_clients = 10
        n_epochs = 5
    else:
        attacks = ['bf', 'lf', 'mimic', 'ipm', 'alie', 'inf']
        aggregators = ['avg', 'krum', 'cm', 'cclip', 'rfa']
        n_clients = 25
        n_epochs = 10

    byzantine_ratio = 0.2
    p = 0.5
    lr = 0.01
    momentum = 0.9

    client_loaders_iid, test_loader = prepare_mnist_data(
        n_clients, 64, iid=True, seed=42)

    all_results = {}
    for agg in aggregators:
        for attack in attacks:
            key = f"{agg}_{attack}"
            print(f"\n  Aggregator={agg}, Attack={attack.upper()}")

            results_for_key = {}
            for algo_name, Algo in [
                ('FedAvg', FedAvg),
                ('FedAvg-M', FedAvgM),
                ('D-Byz-SGDM', DByzSGDM),
                ('Byz-VR-MARINA-PP', ByzVRMARINAPP)
            ]:
                if Algo in [FedAvgM, DByzSGDM]:
                    opt = Algo(CNN_MNIST, n_clients, byzantine_ratio, p,
                               lr, momentum, agg, seed=42)
                elif Algo == ByzVRMARINAPP:
                    opt = Algo(CNN_MNIST, n_clients, byzantine_ratio, p,
                               lr, clip_lambda=10.0, seed=42)
                else:
                    opt = Algo(CNN_MNIST, n_clients, byzantine_ratio, p,
                               lr, agg, seed=42)

                res = run_experiment(
                    opt, CNN_MNIST, client_loaders_iid, test_loader,
                    n_epochs, attack, device, seed=42)
                results_for_key[algo_name] = res
                print(f"    {algo_name}: {res['final_accuracy']:.2f}%")

            all_results[key] = results_for_key

    return all_results


def run_claim5_experiment(device='cpu'):
    from core.models import CNN_MNIST
    from core.algorithms import DByzSGDM, FedAvg, FedAvgM, ByzVRMARINAPP
    from core.utils import prepare_mnist_data

    print("=" * 60)
    print("Claim 5: Non-Byzantine Setting (δ=0)")
    print("=" * 60)

    n_clients = 20
    p = 0.5
    lr = 0.01
    momentum = 0.9
    n_epochs = 10

    print("\n  IID Setting:")
    client_loaders_iid, test_loader = prepare_mnist_data(
        n_clients, 64, iid=True, seed=42)

    opts = {
        'FedAvg': FedAvg(CNN_MNIST, n_clients, 0.0, p, lr, 'avg', seed=42),
        'FedAvg-M': FedAvgM(CNN_MNIST, n_clients, 0.0, p, lr, momentum, 'avg', seed=42),
        'D-Byz-SGDM': DByzSGDM(CNN_MNIST, n_clients, 0.0, p, lr, momentum, 'avg', seed=42),
        'Byz-VR-MARINA-PP': ByzVRMARINAPP(CNN_MNIST, n_clients, 0.0, p, lr, 10.0, seed=42),
    }

    results_iid = {}
    for name, opt in opts.items():
        res = run_experiment(opt, CNN_MNIST, client_loaders_iid, test_loader,
                             n_epochs, 'none', device, seed=42)
        results_iid[name] = res

    print("\n  Non-IID Setting:")
    client_loaders_noniid, _ = prepare_mnist_data(
        n_clients, 64, iid=False, seed=42)

    results_noniid = {}
    for name, opt in opts.items():
        torch.manual_seed(42)
        res = run_experiment(opt, CNN_MNIST, client_loaders_noniid, test_loader,
                             n_epochs, 'none', device, seed=42)
        results_noniid[name] = res

    return {'iid': results_iid, 'noniid': results_noniid}
