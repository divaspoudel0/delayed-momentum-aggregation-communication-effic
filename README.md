# Reproduction: Delayed Momentum Aggregation

Reproduction of **ICML 2026 Paper #756** — *Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation*

- **Paper:** [arXiv:2509.02970](https://arxiv.org/abs/2509.02970)
- **OpenReview:** [KG4CjK6j8Y](https://openreview.net/forum?id=KG4CjK6j8Y)
- **Authors:** Kaoru Otsuka, Yuki Takezawa, Makoto Yamada

## Claims

| Claim | Status |
|---|---|
| 1: D-Byz-SGDM convergence under partial participation (p=0.5, δ=0.2) | ✅ Verified |
| 2: Theorem 4.1 O(cδζ²/p) convergence guarantee | ✅ Verified (numerical audit) |
| 3: Theorem 4.2 Ω(δζ²/p) lower bound tightness | ✅ Verified (numerical audit) |
| 4: MNIST CNN grid (6 attacks × 5 aggregators) | ⚠️ Partially (GPU required for full) |
| 5: Non-Byzantine regularization benefit | ⚠️ Partially (deep net on GPU) |
| 6: Delayed momentum caching mechanism | ✅ Verified |

## Code Structure

```
repro/
├── core/
│   ├── algorithms.py     # D-Byz-SGDM, FedAvg, FedAvg-M, Byz-VR-MARINA-PP
│   ├── aggregators.py    # avg, Krum, cm, cclip, RFA
│   ├── attacks.py        # BF, LF, mimic, IPM, ALIE, INF
│   ├── models.py         # CNN-MNIST, MLP-MNIST, ConvNet-CIFAR10
│   └── utils.py          # Data loading, evaluation
├── experiments/
│   └── runner.py         # Experiment runners
├── repro_claim1.py       # Claim 1 runner
├── repro_claim4.py       # Claim 4 runner
├── repro_claim5.py       # Claim 5 runner
├── repro_claim6.py       # Claim 6 runner
├── repro_theorem_audit.py # Claims 2/3 numerical audit
├── run_quick.py          # Quick verification
├── run_clean.py          # Clean synthetic experiments
├── run_synthetic.py      # Synthetic experiments
├── run_tuned.py          # Tuned synthetic experiments
├── notebook.py           # MoLab notebook
└── requirements.txt      # Dependencies
```

## Quick Start

```bash
pip install -r repro/requirements.txt
cd repro
python3 repro_theorem_audit.py    # Claims 2,3 (seconds)
python3 repro_claim6.py            # Claim 6 (seconds)
python3 run_clean.py               # Claims 1,5 (minutes)
```

## MoLab Notebook

Open `notebook.py` in [MoLab](https://molab.marimo.io) with GPU for full experiments.

## Logbook

Published at: `https://huggingface.co/spaces/divaspoudel0/repro-delayed-momentum-aggregation-communication-efficient-byzantine-robust-federated-learning-w`
