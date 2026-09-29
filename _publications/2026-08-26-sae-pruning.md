---
title: "When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs"
collection: publications
permalink: /publication/sae-pruning
excerpt: 'How post-hoc pruning affects sparse autoencoders (SAEs) trained on LLMs: activation-aware pruning preserves SAE behavior far better than magnitude pruning, middle layers are the most fragile, and a layer-wise sparsity schedule lowers perplexity at the same average sparsity.'
date: 2026-08-26
venue: 'The Conference on Language Modeling (COLM)'
paperurl: 'https://arxiv.org/abs/2608.25941'
citation: 'S. Gupte, X. Zhang, M. Khalili, &quot;When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in LLMs,&quot; <i>The Conference on Language Modeling (COLM)</i>, 2026.'
---

**Suchit Gupte, Xueru Zhang, Mahdi Khalili**

[Paper (arXiv)](https://arxiv.org/abs/2608.25941) &nbsp;·&nbsp; [Code](https://github.com/KhaliliMahdi/sae-robustness-under-pruning) &nbsp;·&nbsp; [BibTeX](#bibtex)

![Layer sensitivity profile of SAEs under pruning](/images/publications/sae-pruning-layer-sensitivity.png)
*Layer sensitivity of Gemma Scope SAEs on gemma-2-2b under pruning, averaged over pruning methods and SAEBench metrics. Middle layers (9–17) degrade the most.*

Abstract
======
Sparse autoencoders (SAEs) are widely used to interpret the internal representations of large language models (LLMs), yet their reliability under post-hoc model compression remains poorly understood. We present a systematic study of how pruning affects SAE behavior and theoretically show that, for a fixed SAE, its impact is governed by perturbation energy, a covariance-weighted norm. This perspective exposes a key limitation of magnitude pruning: by ignoring activation geometry, it distorts the learned representation space and degrades SAE functionality. Activation-aware methods such as Wanda and SparseGPT, in contrast, implicitly control perturbation energy and are therefore substantially more robust at preserving SAE behavior. We further reveal a consistent structural vulnerability across all pruning methods: middle layers are significantly more sensitive to pruning than early or late layers. Guided by this insight, we propose a layer-wise sparsity allocation strategy, achieving lower perplexity under the same average pruning sparsity. Experiments across four model architectures validate our theoretical findings.

Key findings
======
* **Perturbation energy governs SAE degradation.** For a fixed SAE, the damage from pruning a weight matrix W is controlled by ε² = tr(ΔW Σ ΔWᵀ), where Σ is the input activation covariance. Magnitude pruning, Wanda, and SparseGPT minimize successively better approximations of this quantity (Σ replaced by I, then diag(Σ), then the full Σ), which explains the ordering **SparseGPT ≻ Wanda ≻ Magnitude**.
* **Aggregate metrics hide damage.** Magnitude-pruned middle layers keep KL-divergence scores ≥ 0.96 while intervention metrics (SCR, TPP) collapse, so reconstruction fidelity alone cannot certify that an SAE is still reliable.
* **Middle layers are the most fragile** across all pruning methods, metrics, and sparsity levels, driven by perturbation accumulated from upstream layers.
* **Layer-wise sparsity allocation.** A ramp schedule that protects early layers (20% sparsity rising to 60% at the middle layers) lowers perplexity at the same 50% average sparsity.

Results
======
![SAEBench metrics under pruning, per layer](/images/publications/sae-pruning-metrics.png)
*Percent change from the dense baseline for eight SAEBench metrics across all 26 layers of gemma-2-2b at 50% sparsity. Magnitude pruning (red) degrades SAEs far more than the activation-aware methods.*

Perplexity of gemma-2-2b at 50% average sparsity:

| Method | Uniform sparsity | Layer-wise sparsity |
|---|---|---|
| Wanda | 212 | **148** |
| SparseGPT | 141 | **98** |

The findings hold across four model and SAE pairs: pythia-70m (SAELens), gemma-2-2b and gemma-2-9b (Gemma Scope), and mistral-7b.

BibTeX
======
{% raw %}
```bibtex
@inproceedings{gupte2026pruning,
  title     = {When Pruning Meets Interpretability: Preserving Sparse Autoencoder Robustness in {LLM}s},
  author    = {Gupte, Suchit and Zhang, Xueru and Khalili, Mohammad Mahdi},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
{% endraw %}
