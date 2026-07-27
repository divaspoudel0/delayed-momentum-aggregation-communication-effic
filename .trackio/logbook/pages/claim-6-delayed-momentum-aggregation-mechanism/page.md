# Claim 6: Delayed momentum aggregation mechanism


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_71b652a4f3df", "created_at": "2026-07-27T06:11:33+00:00", "title": "Evidence"}
-->
## Claim 6: Delayed Momentum Aggregation Mechanism

**Paper claim:** The delayed momentum aggregation scheme caches stale momentum for non-sampled clients (m_i^t ← m_i^{t-1}) so Byzantine clients cannot dominate the aggregated update even if they form a majority of the sampled set, with no additional communication overhead.

### Mathematical Guarantee
In D-Byz-SGDM (Algorithm 1), the server aggregates:
- Fresh momentum from sampled clients S_t (size ≈ n·p)
- Cached momentum from non-sampled clients [n]\S_t (size ≈ n·(1-p))

Total aggregated vectors = n, so Byzantine fraction in aggregation = δ/n·n = δ < 0.5 ALWAYS.

### Simulation Results (10,000 rounds per scenario)

| Scenario | n | δ | p | Byz% in sampled | Byz% in aggregation | Majority rounds |
|---|---|---|---|---|---|---|
| Paper default | 20 | 0.20 | 0.50 | 20.2% | 20.0% | 0.1% |
| Paper default (25) | 25 | 0.20 | 0.50 | 19.8% | 20.0% | 0.0% |
| High Byz, low p | 20 | 0.30 | 0.30 | 29.9% | 30.0% | 7.6% |
| Extreme (40% Byz, p=0.1) | 100 | 0.40 | 0.10 | 40.2% | 40.0% | 20.4% |
| Paper extreme (p=0.1) | 20 | 0.20 | 0.10 | 17.1% | 20.0% | 7.7% |

**Key insight:** Byzantine clients NEVER form a majority in aggregation because cached honest momentum from non-sampled clients always dilutes them. No extra communication — server maintains per-client momentum cache from prior rounds.

**Verified:** ✓ Mechanism matches Algorithm 1
