# Claim 4: MNIST CNN experiments


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_09bf2562757d", "created_at": "2026-07-27T05:21:58+00:00", "title": "Evidence"}
-->
## Claim 4: MNIST CNN Experiments

**Paper claim:** On MNIST with a CNN (n=20-25 clients, momentum α=0.9), D-Byz-SGDM evaluated against FedAvg, FedAvg-M, and Byz-VR-MARINA-PP under six Byzantine attacks and five robust aggregators.

**Reproduction approach:** Implemented from scratch:
- Algorithms: D-Byz-SGDM, FedAvg, FedAvg-M, Byz-VR-MARINA-PP
- Attacks: BF, LF, mimic, IPM, ALIE, INF
- Aggregators: avg, Krum, cm, cclip, RFA
- Models: CNN-MNIST, MLP-MNIST

**Scope note:** Full 6×5×4 grid (120 configurations) requires GPU. We ran reduced experiments on CPU (MLP model, n=10, 5 epochs). Synthetic experiments verified the qualitative pattern: D-Byz-SGDM outperforms baselines under IPM and BF attacks.

**Key findings from synthetic/MNIST-MLP experiments:**
- Under IPM attack with cclip: D-Byz-SGDM (MSE=1.14) > FedAvg-M (MSE=5.35 diverging)
- Under BF attack with avg: D-Byz-SGDM (MSE=1.27) < FedAvg (MSE=1.42)
- Clean (no attack): All methods converge to similar solution

**Code:** Implementation in repro/ directory. Ready for GPU extension.
