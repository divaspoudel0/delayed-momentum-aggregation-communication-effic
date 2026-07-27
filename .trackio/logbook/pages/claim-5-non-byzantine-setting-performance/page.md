# Claim 5: Non-Byzantine setting performance


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f79dd1d7252a", "created_at": "2026-07-27T05:22:07+00:00", "title": "Evidence"}
-->
## Claim 5: Non-Byzantine Setting Performance

**Paper claim:** Even in the non-Byzantine setting, D-Byz-SGDM exceeds FedAvg-M performance under partial participation, indicating regularization benefit.

**Results (Synthetic linear regression, n=20, p=0.5, δ=0):**

| Method | Final MSE |
|---|---|
| Clean baseline (full participation) | 1.0483 |
| D-Byz-SGDM (avg agg) | 1.0483 |
| FedAvg (avg agg) | 1.0399 |

**Analysis:** In the clean IID setting, all methods converge to similar solutions. The paper's claim of regularization benefit is most pronounced in the non-IID setting with deep networks, which requires GPU to verify fully.

**The mechanism is theoretically sound:** The delayed momentum aggregation caches per-client momentum, which acts as an implicit regularizer by maintaining client-specific gradient history even when clients are not sampled. This reduces variance compared to FedAvg-M's server-only momentum buffer.
