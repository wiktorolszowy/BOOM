# Compute-time comparison: Chemprop vs. ElasticNet tuning

We believe a comparison between models should also refer to wall-clock differences, which is why we compared the wall-clock time
Chemprop and ElasticNet spend on hyperparameter tuning, on the same hardware
(single NVIDIA A10G GPU, 32 CPU cores).

- **Chemprop tuning**: 50 independently-trained MPNNs per endpoint, randomly
  sampling message-passing depth (2–6), dropout (0–0.40), FFN layers (1–3),
  and hidden size (300–2400), each trained on GPU for up to 100 epochs with
  early stopping.
- **ElasticNet tuning**: a single `ElasticNetCV` call per endpoint, which
  internally cross-validates a grid of 4 l1_ratios × 30 alphas × 3 folds,
  runs entirely on CPU.

| Endpoint | Chemprop with tuning | ElasticNet with tuning | Speed ratio |
|---|---|---|---|
| HoF | 4h38m | 23s | 726× |
| Density | 3h48m | 2m36s | 88× |
| HOMO | 29h37m | 5m10s | 344× |

(For Chemprop, only these three endpoints had completed their full 50-run tuning search; other endpoints were partially completed or pending at the time of analysis).

Chemprop's tuning cost is also highly variable *within* a single endpoint:
because the sampled hidden size spans an 8× range (300–2400), the resulting
networks range from roughly 300K to 29M parameters, and the largest networks
take 3.4–6.5× longer per run than the smallest ones on the same dataset.

On top of that, the comparison above understates the real gap in
compute resources: ElasticNet's tuning never touches the GPU at all, so the
A10G sat completely idle while it ran. If the cost is measured in terms of
the specialized (and typically scarcer/more expensive) hardware each method
actually occupies, rather than plain wall-clock time, the true difference
between the two must be even larger than the 88–726× shown here.

There is also likely a large inference-time gap between the two, driven
by model size: Chemprop's forward pass scales with its (tuned) parameter
count, which can reach tens of millions, whereas ElasticNet's `.predict()` is
a single, cheap matrix multiply.

**Implication for QSPR practice:** at a fixed compute budget, simple models
like ElasticNet or XGBoost can be easily tuned and then applied to far more
compounds in a virtual screening campaign than a heavily-tuned GNN. Importantly, simple models
are considerably easier to interpret. We believe that given these practical advantages,
complex models should have to demonstrate a clear, consistent improvement in
OOD generalization over such simple baselines before they can be justified as
the standard modeling approach for QSPR.
