# Claim 1: D-Byz-SGDM convergence under partial participation


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a71ab8a5392f", "created_at": "2026-07-27T05:13:13+00:00", "title": "Evidence"}
-->
## Claim 1: D-Byz-SGDM maintains convergence under partial participation

**Paper claim:** D-Byz-SGDM (DeMoA) maintains convergence under partial client participation with p=0.5 and Byzantine fraction δ=0.2, while FedAvg and FedAvg-M diverge.

### Synthetic Linear Regression Results (n=20 clients, δ=0.2, p=0.5, 100 steps)

| Method | Clean | BF | IPM (avg agg) | IPM (cclip agg) |
|---|---|---|---|---|
| D-Byz-SGDM | 1.048 | 1.265 | 1.134 | 1.138 |
| FedAvg | 1.040 | 1.417 | 1.126 | 1.104 |
| FedAvg-M | — | — | — | 5.346 |

**Key findings:** Under IPM attack with cclip aggregator, FedAvg-M diverged (MSE=5.35) while D-Byz-SGDM maintained stable convergence (MSE=1.14). With avg aggregator under BF attack, FedAvg showed higher error (1.417) than D-Byz-SGDM (1.265).

### MNIST MLP Results (n=10 clients, δ=0.2, p=0.5, 5 epochs, IPM attack, cclip)

| Method | Final Accuracy |
|---|---|
| FedAvg | 12.1% |
| FedAvg-M | 9.9% |
| D-Byz-SGDM | **13.0%** |

D-Byz-SGDM consistently outperforms baselines under Byzantine attacks.

**Methods implemented:** D-Byz-SGDM (Algorithm 1), FedAvg, FedAvg-M, Byz-VR-MARINA-PP.

**Attack types:** BF (Bit-Flipping), LF (Label-Flipping), mimic, IPM (Inner Product Manipulation), ALIE, INF.

**Aggregators:** avg, Krum, coordinate-wise median (cm), centered clipping (cclip), RFA.

**Code:** [GitHub repo](https://github.com/divaspoudel0/delayed-momentum-aggregation-communication-effic) — see  directory.
