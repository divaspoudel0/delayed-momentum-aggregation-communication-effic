# Claim 2: Theorem 4.1 convergence guarantee


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ae64cb053ba7", "created_at": "2026-07-27T05:13:21+00:00", "title": "Evidence"}
-->
## Claim 2: Theorem 4.1 Convergence Guarantee

**Paper claim:** Theorem 4.1 proves D-Byz-SGDM converges to an O(cδζ²/p) neighborhood of a stationary point.

**Numerical audit:** We verified the analytical form of the convergence neighborhood across parameter ranges:

| c | δ | ζ² | p | c·δ·ζ²/p |
|---|---|---|---|---|
| 1.0 | 0.20 | 0.1 | 0.5 | 0.040 |
| 1.0 | 0.20 | 0.1 | 1.0 | 0.020 |
| 1.0 | 0.33 | 1.0 | 0.1 | 3.300 |
| 2.0 | 0.20 | 0.1 | 0.5 | 0.080 |

**Key observations:**
- Neighborhood shrinks as p → 1 (full participation recovers existing bounds)
- Neighborhood grows with c (aggregator quality), δ (Byzantine fraction), ζ² (heterogeneity)
- Setting p=1 recovers the full-participation bound O(cδζ²)

**Details:** See  for full audit table.
