---
title: "An Efficient Training Algorithm for Models with Block-wise Sparsity"
collection: publications
permalink: /publication/blockwise-sparsity
excerpt: 'A Kronecker-product parameterization for training block-wise sparse weight matrices without starting from a dense model: it cuts training parameters and FLOPs, and it can choose the block size within a single training run.'
date: 2025-03-27
venue: 'Transactions on Machine Learning Research (TMLR)'
paperurl: 'https://arxiv.org/abs/2503.21928'
citation: 'D. Zhu, Z. Zuo, M. Khalili, &quot;An Efficient Training Algorithm for Models with Block-wise Sparsity,&quot; <i>Transactions on Machine Learning Research (TMLR)</i>, 2025.'
---

**Ding Zhu, Zhiqun Zuo, Mahdi Khalili**

[Paper (arXiv)](https://arxiv.org/abs/2503.21928) &nbsp;·&nbsp; [Code](https://github.com/KhaliliMahdi/kronvit) &nbsp;·&nbsp; [Tutorial: wall time and memory](/posts/2026/10/kronecker-layer-benchmark/) &nbsp;·&nbsp; [BibTeX](#bibtex)

![Kronecker-product parameterization of a block-wise sparse matrix](/images/publications/blockwise-sparsity-kronecker.png)
*A weight matrix is written as (S ⊙ A<sub>i</sub>) ⊗ B<sub>i</sub>. When the small matrix S is sparse, the product is block-wise sparse, and the block size equals the size of B<sub>i</sub>. White entries are zero.*

Abstract
======
Large-scale machine learning (ML) models are increasingly being used in critical domains like education, lending, recruitment, healthcare, criminal justice, etc. However, the training, deployment, and utilization of these models demand substantial computational resources. To decrease computation and memory costs, machine learning models with sparse weight matrices are widely used in the literature. Among sparse models, those with special sparse structures (e.g., models with block-wise sparse weight matrices) fit better with the hardware accelerators and can decrease the memory and computation costs during the inference. Unfortunately, while there are several efficient training methods, none of them are designed to train a block-wise sparse model efficiently. As a result, the current methods for training block-wise sparse models start with full and dense models leading to inefficient training. In this work, we focus on training models with block-wise sparse matrices and propose an efficient training algorithm to decrease both computation and memory costs during training and inference. In addition, we will show that our proposed method enables us to efficiently find the right block size for the sparsity pattern during the training process. Our extensive empirical and theoretical analyses show that our algorithms can decrease the computation and memory costs significantly without a performance drop compared to baselines.

Key findings
======
* **Block-wise sparsity through a Kronecker factorization.** Each weight matrix is parameterized as W = Σ<sub>i=1</sub><sup>r</sup> (S ⊙ A<sub>i</sub>) ⊗ B<sub>i</sub>, with an ℓ1 penalty on S. A sparse S makes W block-wise sparse, with blocks the size of B<sub>i</sub>. Any block-wise sparse matrix with equal-sized blocks can be written in this form (Proposition 1).
* **Lower training cost.** The factors are trained directly, so training never touches a dense matrix. Group LASSO and pruning, by contrast, start from the full model. For an 8 × 256 matrix with rank 1, the factorization has 128 trainable parameters instead of 2,048. For ViT-tiny on CIFAR-100, it trains 0.16M parameters instead of 5.5M, with 65.37M training FLOPs instead of 2.16G.
* **Block size chosen in one training run.** Each candidate block size gets its own set of factors, and a group penalty on each set pushes all but one pattern to zero during training. Group LASSO and pruning need one run per candidate. On ViT-tiny, the run kept the 2 × 2 pattern. Trained separately, the three candidate patterns reach 64.56 (2 × 2), 62.99 (4 × 4), and 62.88 (8 × 8) accuracy.
* **Accuracy compared with baselines.** On CIFAR-100 with 4 × 4 blocks, the method is more accurate than group LASSO, elastic group LASSO, and block-wise RigL on ViT-tiny and Swin-tiny, and than both group LASSO variants on ViT-base (no block-wise RigL result is reported for ViT-base). It is 1.3–3.9 points below the dense models. On a linear MNIST model it is the most accurate method at 2 × 2 blocks, but block-wise RigL is more accurate at larger block sizes.

Results
======
CIFAR-100, 4 × 4 blocks (Table 3 of the paper):

| Model | Method | Accuracy | Sparsity (%) | Training params | Training FLOPs |
|---|---|---|---|---|---|
| ViT-tiny | Dense | 64.32 | – | 5.5M | 2.16G |
| ViT-tiny | Group LASSO | 60.41 | 49.99 | 5.5M | 2.16G |
| ViT-tiny | Elastic group LASSO | 61.92 | 49.92 | 5.5M | 2.16G |
| ViT-tiny | Block-wise RigL | 49.56 | 50.67 | 5.5M | 2.16G |
| ViT-tiny | **Ours** | **62.99** | 49.64 | **0.16M** | **65.37M** |
| Swin-tiny | Dense | 81.44 | – | 27.60M | 26.18G |
| Swin-tiny | Group LASSO | 75.87 | 50.24 | 27.60M | 26.18G |
| Swin-tiny | Elastic group LASSO | 76.34 | 50.19 | 27.60M | 26.18G |
| Swin-tiny | Block-wise RigL | 60.30 | 50.03 | 27.60M | 26.18G |
| Swin-tiny | **Ours** | **77.54** | 53.25 | **5.3M** | **167.33M** |

Bold marks the best result among the sparse methods. On ViT-base, the method trains 11.58M parameters instead of 87.34M and reaches 69.82 accuracy. That is below the dense model (71.34) and above group LASSO (68.41) and elastic group LASSO (66.95).

BibTeX
======
{% raw %}
```bibtex
@article{zhu2025blockwise,
  title   = {An Efficient Training Algorithm for Models with Block-wise Sparsity},
  author  = {Zhu, Ding and Zuo, Zhiqun and Khalili, Mohammad Mahdi},
  journal = {Transactions on Machine Learning Research},
  year    = {2025}
}
```
{% endraw %}
