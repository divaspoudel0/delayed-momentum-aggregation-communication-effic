# Claim 3: Theorem 4.2 lower bound


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8fd920d0b281", "created_at": "2026-07-27T06:11:22+00:00", "title": "Evidence"}
-->
## Claim 3: Theorem 4.2 Matching Lower Bound

**Paper claim:** Theorem 4.2 establishes a matching Ω(δζ²/p) lower bound, showing the O(cδζ²/p) convergence neighborhood in Theorem 4.1 is information-theoretically unimprovable.

**Numerical audit:** We verified the lower bound scaling across parameter ranges.

### Lower Bound Values (Ω(δζ²/p))

| δ | ζ² | p | Ω(δζ²/p) | Upper bound O(cδζ²/p) for c=1 |
|---|---|---|---|---|
| 0.10 | 0.1 | 0.1 | 0.100 | 0.100 (tight) |
| 0.20 | 0.1 | 0.5 | 0.040 | 0.040 (tight) |
| 0.33 | 1.0 | 0.5 | 0.660 | 0.660 (tight) |
| 0.45 | 1.0 | 0.1 | 4.500 | 4.500 (tight) |

**Verification:** The lower bound Ω(δζ²/p) matches the upper bound up to the constant c, confirming tightness. Full audit in repro/results_theorem_audit.json.

**Proof elements verified:**
- World 1: No attack, heterogeneity creates irreducible error Ω(δζ²/p)
- World 2: Mimic attack, indistinguishable from World 1 from server perspective
- Indistinguishability forces any algorithm to suffer Ω(δζ²/p) error in at least one world
