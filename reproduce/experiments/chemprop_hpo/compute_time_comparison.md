# Compute-time comparison: Chemprop vs. ElasticNet tuning

This note compares how much wall-clock compute Chemprop and ElasticNet each
spend on hyperparameter tuning, on the same hardware (single NVIDIA A10G GPU,
32 CPU cores). Numbers are taken directly from existing run logs via
[`compute_time_comparison.py`](compute_time_comparison.py) — no additional
training was run to produce them.

- **Chemprop tuning**: 50 independently-trained MPNNs per endpoint, randomly
  sampling message-passing depth (2–6), dropout (0–0.40), FFN layers (1–3),
  and hidden size (300–2400), each trained on GPU for up to 100 epochs with
  early stopping.
- **ElasticNet tuning**: a single `ElasticNetCV` call per endpoint, which
  internally cross-validates a grid of 4 l1_ratios × 30 alphas × 3 folds,
  run entirely on CPU.

| Endpoint | Chemprop tuning (50 runs) | ElasticNet tuning (1 call) | Speed ratio |
|---|---|---|---|
| HoF | 4h38m | 23s | 726× |
| Density | 3h48m | 2m36s | 88× |
| HOMO | 29h37m | 5m10s | 344× |

(Only these three endpoints had completed their full 50-run tuning search
before the study was stopped; the remaining seven were partially or not yet
tuned.)

Chemprop's tuning cost is also highly variable *within* a single endpoint:
because the sampled hidden size spans an 8× range (300–2400), the resulting
networks range from roughly 300K to 29M parameters, and the largest networks
take 3.4–6.5× longer per run than the smallest ones on the same dataset.

On top of that, the comparison above likely understates the real gap in
compute resources: ElasticNet's tuning never touches the GPU at all, so the
A10G sat completely idle while it ran. If the cost is measured in terms of
the specialized (and typically scarcer/more expensive) hardware each method
actually occupies, rather than plain wall-clock time, the true difference
between the two is probably even larger than the 88–726× shown here.

**Implication for QSPR practice:** at a fixed compute budget, simple models
like ElasticNet or XGBoost can be tuned and then applied to far more
compounds in a virtual screening campaign than a heavily-tuned GNN, and they
are considerably easier to interpret. Given these practical advantages,
complex models should have to demonstrate a clear, consistent improvement in
OOD generalization over such simple baselines before they can be justified as
the standard modeling approach for QSPR.
