import torch
import copy
from .aggregators import robust_aggregator

class FederatedOptimizer:
    def __init__(self, model_fn, n_clients, byzantine_ratio=0.0,
                 participation_prob=1.0, seed=42):
        self.model_fn = model_fn
        self.n_clients = n_clients
        self.byzantine_ratio = byzantine_ratio
        self.participation_prob = participation_prob
        self.seed = seed
        self.n_byzantine = int(n_clients * byzantine_ratio)
        self.n_honest = n_clients - self.n_byzantine
        self.byzantine_ids = set(range(self.n_honest, n_clients))
        self.rng = torch.Generator().manual_seed(seed)

    def sample_clients(self):
        mask = torch.rand(self.n_clients, generator=self.rng) < self.participation_prob
        return set(torch.where(mask)[0].tolist())

    def is_byzantine(self, client_id):
        return client_id in self.byzantine_ids


class DByzSGDM(FederatedOptimizer):
    def __init__(self, model_fn, n_clients, byzantine_ratio=0.0,
                 participation_prob=1.0, lr=0.01, momentum=0.9,
                 agg_type='avg', seed=42):
        super().__init__(model_fn, n_clients, byzantine_ratio,
                         participation_prob, seed)
        self.lr = lr
        self.momentum = momentum
        self.agg_type = agg_type
        self.client_momentum = {}

    def init_state(self, global_model):
        init_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])
        for i in range(self.n_clients):
            self.client_momentum[i] = init_params.clone()

    def step(self, global_model, client_data_loaders, attack_type='none'):
        sampled = self.sample_clients()
        state_dict = copy.deepcopy(global_model.state_dict())
        all_params = torch.cat([p.data.view(-1) for p in global_model.parameters()])

        momentum_list = []
        for i in range(self.n_clients):
            if i in sampled:
                model_i = self.model_fn()
                model_i.load_state_dict(state_dict)
                model_i.train()
                data_iter = iter(client_data_loaders[i])
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    inputs, targets = None, None
                if inputs is not None:
                    optimizer = torch.optim.SGD(model_i.parameters(), lr=1.0)
                    optimizer.zero_grad()
                    outputs = model_i(inputs)
                    loss = torch.nn.functional.nll_loss(outputs, targets)
                    loss.backward()
                    grad = torch.cat([p.grad.data.view(-1) for p in model_i.parameters()])
                    momentum_i = (1 - self.momentum) * self.client_momentum[i] + self.momentum * grad
                else:
                    momentum_i = self.client_momentum[i]
            else:
                momentum_i = self.client_momentum[i]

            if self.is_byzantine(i):
                momentum_i = self.apply_attack(momentum_i, attack_type, all_params, momentum_list)
            momentum_list.append(momentum_i)
            self.client_momentum[i] = momentum_i

        agg_momentum = robust_aggregator(momentum_list, self.agg_type)

        offset = 0
        for p in global_model.parameters():
            n_params = p.data.numel()
            delta = agg_momentum[offset:offset + n_params].view(p.data.shape)
            p.data -= self.lr * delta
            offset += n_params

        return sampled

    @staticmethod
    def apply_attack(momentum, attack_type, ref_params, all_momentum):
        if attack_type == 'none':
            return momentum
        elif attack_type == 'bf':
            return -momentum
        elif attack_type == 'ipm':
            if all_momentum:
                honest = [m for m in all_momentum]
                if honest:
                    avg = torch.stack(honest).mean(dim=0)
                    return -avg * 1.0
            return -momentum
        elif attack_type == 'alie':
            if len(all_momentum) >= 2:
                stacked = torch.stack(all_momentum)
                mu = stacked.mean(dim=0)
                sigma = stacked.std(dim=0)
                return mu - 1.0 * sigma
            return momentum
        elif attack_type == 'inf':
            return 100.0 * torch.ones_like(momentum) * torch.norm(momentum)
        elif attack_type == 'mimic':
            if all_momentum:
                honest = [m for m in all_momentum]
                if honest:
                    avg = torch.stack(honest).mean(dim=0)
                    noise = 0.1 * torch.randn_like(avg) * torch.norm(avg)
                    return avg + noise
            return momentum
        return momentum


class FedAvg(FederatedOptimizer):
    def __init__(self, model_fn, n_clients, byzantine_ratio=0.0,
                 participation_prob=1.0, lr=0.01, agg_type='avg', seed=42):
        super().__init__(model_fn, n_clients, byzantine_ratio,
                         participation_prob, seed)
        self.lr = lr
        self.agg_type = agg_type

    def step(self, global_model, client_data_loaders, attack_type='none'):
        sampled = self.sample_clients()
        state_dict = copy.deepcopy(global_model.state_dict())
        gradients = []

        for i in range(self.n_clients):
            if i not in sampled:
                continue
            model_i = self.model_fn()
            model_i.load_state_dict(state_dict)
            model_i.train()
            data_iter = iter(client_data_loaders[i])
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                continue
            optimizer = torch.optim.SGD(model_i.parameters(), lr=1.0)
            optimizer.zero_grad()
            outputs = model_i(inputs)
            loss = torch.nn.functional.nll_loss(outputs, targets)
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in model_i.parameters()])

            if self.is_byzantine(i):
                if attack_type == 'bf':
                    grad = -grad
                elif attack_type == 'ipm':
                    grad = -grad * 1.0
                elif attack_type == 'inf':
                    grad = 100.0 * torch.ones_like(grad) * torch.norm(grad)

            gradients.append(grad)

        if not gradients:
            return sampled

        agg_grad = robust_aggregator(gradients, self.agg_type)

        offset = 0
        for p in global_model.parameters():
            n_params = p.data.numel()
            delta = agg_grad[offset:offset + n_params].view(p.data.shape)
            p.data -= self.lr * delta
            offset += n_params

        return sampled


class FedAvgM(FederatedOptimizer):
    def __init__(self, model_fn, n_clients, byzantine_ratio=0.0,
                 participation_prob=1.0, lr=0.01, momentum=0.9,
                 agg_type='avg', seed=42):
        super().__init__(model_fn, n_clients, byzantine_ratio,
                         participation_prob, seed)
        self.lr = lr
        self.momentum = momentum
        self.agg_type = agg_type
        self.server_momentum = None

    def step(self, global_model, client_data_loaders, attack_type='none'):
        sampled = self.sample_clients()
        state_dict = copy.deepcopy(global_model.state_dict())
        gradients = []

        for i in range(self.n_clients):
            if i not in sampled:
                continue
            model_i = self.model_fn()
            model_i.load_state_dict(state_dict)
            model_i.train()
            data_iter = iter(client_data_loaders[i])
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                continue
            optimizer = torch.optim.SGD(model_i.parameters(), lr=1.0)
            optimizer.zero_grad()
            outputs = model_i(inputs)
            loss = torch.nn.functional.nll_loss(outputs, targets)
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in model_i.parameters()])

            if self.is_byzantine(i):
                if attack_type == 'bf':
                    grad = -grad
                elif attack_type == 'ipm':
                    grad = -grad * 1.0
                elif attack_type == 'inf':
                    grad = 100.0 * torch.ones_like(grad) * torch.norm(grad)

            gradients.append(grad)

        if not gradients:
            return sampled

        agg_grad = robust_aggregator(gradients, self.agg_type)

        if self.server_momentum is None:
            self.server_momentum = agg_grad.clone()
        self.server_momentum = self.momentum * self.server_momentum + (1 - self.momentum) * agg_grad

        offset = 0
        for p in global_model.parameters():
            n_params = p.data.numel()
            delta = self.server_momentum[offset:offset + n_params].view(p.data.shape)
            p.data -= self.lr * delta
            offset += n_params

        return sampled


class ByzVRMARINAPP(FederatedOptimizer):
    def __init__(self, model_fn, n_clients, byzantine_ratio=0.0,
                 participation_prob=1.0, lr=0.01, clip_lambda=10.0,
                 seed=42):
        super().__init__(model_fn, n_clients, byzantine_ratio,
                         participation_prob, seed)
        self.lr = lr
        self.clip_lambda = clip_lambda
        self.prev_gradients = {}
        self.last_full_grad = None
        self.step_counter = 0
        self.full_freq = 10

    def step(self, global_model, client_data_loaders, attack_type='none'):
        self.step_counter += 1
        sampled = self.sample_clients()
        state_dict = copy.deepcopy(global_model.state_dict())
        gradients = []

        for i in range(self.n_clients):
            if i not in sampled:
                continue
            model_i = self.model_fn()
            model_i.load_state_dict(state_dict)
            model_i.train()
            data_iter = iter(client_data_loaders[i])
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                continue
            optimizer = torch.optim.SGD(model_i.parameters(), lr=1.0)
            optimizer.zero_grad()
            outputs = model_i(inputs)
            loss = torch.nn.functional.nll_loss(outputs, targets)
            loss.backward()
            grad = torch.cat([p.grad.data.view(-1) for p in model_i.parameters()])

            if self.is_byzantine(i):
                if attack_type == 'bf':
                    grad = -grad
                elif attack_type == 'ipm':
                    grad = -grad * 1.0

            if i in self.prev_gradients:
                diff = grad - self.prev_gradients[i]
                clip_norm = torch.norm(diff)
                if clip_norm > self.clip_lambda:
                    diff = diff / clip_norm * self.clip_lambda
                grad = self.prev_gradients[i] + diff

            self.prev_gradients[i] = grad
            gradients.append(grad)

        if not gradients:
            return sampled

        agg_grad = torch.stack(gradients).mean(dim=0)

        offset = 0
        for p in global_model.parameters():
            n_params = p.data.numel()
            delta = agg_grad[offset:offset + n_params].view(p.data.shape)
            p.data -= self.lr * delta
            offset += n_params

        return sampled
