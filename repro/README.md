# Reproduction: Delayed Momentum Aggregation

Reproduction of ICML 2026 Paper #756 — *Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation*

**arXiv:** [2509.02970](https://arxiv.org/abs/2509.02970)  
**OpenReview:** [KG4CjK6j8Y](https://openreview.net/forum?id=KG4CjK6j8Y)  
**Authors:** Kaoru Otsuka, Yuki Takezawa, Makoto Yamada

## Structure

- `core/` — Core implementations (algorithms, attacks, aggregators, models)
- `experiments/` — Experiment scripts for each claim
- `repro_claim*.py` — Entry point scripts
- `notebook.ipynb` — Jupyter notebook for easy reproduction

## Setup

```bash
pip install -r requirements.txt
python repro_claim1.py
```

## Claims

1. **Claim 1**: D-Byz-SGDM maintains convergence under partial participation (p=0.5, δ=0.2)
2. **Claim 2**: Theorem 4.1 convergence guarantee O(cδζ²/p)
3. **Claim 3**: Theorem 4.2 matching Ω(δζ²/p) lower bound
4. **Claim 4**: MNIST CNN experiments with 6 attacks × 5 aggregators
5. **Claim 5**: Non-Byzantine setting, D-Byz-SGDM exceeds FedAvg-M
6. **Claim 6**: Delayed momentum aggregation mechanism (Algorithm 1)
