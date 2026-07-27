import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np

def prepare_mnist_data(n_clients, batch_size=64, iid=True, seed=42):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    n_samples = len(train_dataset)
    indices = np.arange(n_samples)

    if iid:
        rng.shuffle(indices)
        client_data = np.array_split(indices, n_clients)
    else:
        labels = np.array([train_dataset[i][1] for i in range(n_samples)])
        sorted_idx = np.argsort(labels)
        sorted_by_label = [sorted_idx[labels[sorted_idx] == c] for c in range(10)]
        clients_data = []
        for i in range(n_clients):
            client_idxs = []
            for c in range(10):
                n_per_client = len(sorted_by_label[c]) // n_clients
                start = i * n_per_client
                end = (i + 1) * n_per_client if i < n_clients - 1 else len(sorted_by_label[c])
                client_idxs.extend(sorted_by_label[c][start:end].tolist())
            clients_data.append(np.array(client_idxs))
        client_data = clients_data

    client_loaders = []
    for c_idxs in client_data:
        subset = Subset(train_dataset, c_idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return client_loaders, test_loader


def prepare_cifar10_data(n_clients, batch_size=64, iid=True, seed=42):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    rng = np.random.RandomState(seed)
    n_samples = len(train_dataset)
    indices = np.arange(n_samples)

    if iid:
        rng.shuffle(indices)
        client_data = np.array_split(indices, n_clients)
    else:
        labels = np.array([train_dataset[i][1] for i in range(n_samples)])
        sorted_idx = np.argsort(labels)
        sorted_by_label = [sorted_idx[labels[sorted_idx] == c] for c in range(10)]
        clients_data = []
        for i in range(n_clients):
            client_idxs = []
            for c in range(10):
                n_per_client = len(sorted_by_label[c]) // n_clients
                start = i * n_per_client
                end = (i + 1) * n_per_client if i < n_clients - 1 else len(sorted_by_label[c])
                client_idxs.extend(sorted_by_label[c][start:end].tolist())
            clients_data.append(np.array(client_idxs))
        client_data = clients_data

    client_loaders = []
    for c_idxs in client_data:
        subset = Subset(train_dataset, c_idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return client_loaders, test_loader


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total


def compute_loss(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = F.nll_loss(outputs, targets, reduction='sum')
            total_loss += loss.item()
            total_samples += targets.size(0)
    return total_loss / total_samples
