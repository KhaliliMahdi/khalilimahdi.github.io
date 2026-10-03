---
title: 'Kronecker-factored layers in PyTorch: wall time and GPU memory'
date: 2026-10-03
permalink: /posts/2026/10/kronecker-layer-benchmark/
categories:
  - tutorials
excerpt: 'Three PyTorch implementations of the block-wise sparse Kronecker layer, compared with dense and group LASSO baselines: wall time and peak GPU memory per training step at batch 256 and 20,000.'
tags:
  - block-wise sparsity
  - Kronecker product
  - PyTorch
---

In [our TMLR paper](/publication/blockwise-sparsity), each weight matrix is written as a sum of Kronecker products,

$$W = \sum_{i=1}^{r} (S \odot A_i) \otimes B_i,$$

with an $$\ell_1$$ penalty on $$S$$. When $$S$$ is sparse, $$W$$ is block-wise sparse, and the block size equals the size of $$B_i$$. The model trains far fewer parameters than a dense layer. This tutorial asks a practical question: **does a training step actually run faster, and use less GPU memory, than a dense layer?** We write the layer in three ways, compare it with two baselines, and time everything on one GPU.

The full script is [`benchmarks/kron_simple_benchmark.py`](https://github.com/KhaliliMahdi/kronvit/blob/master/benchmarks/kron_simple_benchmark.py) in the [kronvit repository](https://github.com/KhaliliMahdi/kronvit). It is a single file and needs only PyTorch.

Setup
======
* **Network.** A 3-layer MLP: 1024 → 1024 → 1024 → 10, with ReLU. In the Kronecker models, the two 1024 × 1024 hidden layers are Kronecker layers. The 10-class output layer stays dense in every model, because 10 does not split into blocks.
* **Training step.** Forward pass, cross-entropy loss plus the model's penalty, backward pass, and an Adam update. Inputs and labels are random, because the speed of a step does not depend on the data.
* **Measurement.** 10 untimed warm-up steps, then 100 timed steps. **Wall time** is the average time of one training step (one mini-batch), from the CPU clock, after `torch.cuda.synchronize()`. **Peak memory** is `torch.cuda.max_memory_allocated()` during the timed steps.
* **Hardware and precision.** One NVIDIA RTX A6000, PyTorch 2.8, everything in fp32 (no mixed precision, no TF32).
* **Batch size.** "Batch" here is the number of rows that go into each linear layer. In a vision transformer, every image contributes one row per token. For example, DeiT-tiny at 224 × 224 has 197 tokens per image, so a batch of 20,000 rows is about 100 images.

Rank $$r$$ and block size $$m$$ (each $$B_i$$ is $$m \times m$$) are the two knobs. For a 1024 × 1024 layer, $$S$$ and each $$A_i$$ are $$(1024/m) \times (1024/m)$$.

The baselines
======
**Dense** is three `nn.Linear` layers, with no regularizer.

**Dense + group LASSO** is the usual way to get block-wise sparsity from a dense model. It splits each hidden weight matrix into $$m \times m$$ blocks and adds the sum of the blocks' Frobenius norms to the loss. This pushes whole blocks to zero, the same structure that a sparse $$S$$ gives:

```python
def group_lasso(weight, block):
    out_f, in_f = weight.shape
    blocks = weight.view(out_f // block, block, in_f // block, block)
    norms = (blocks.pow(2).sum(dim=(1, 3)) + 1e-12).sqrt()  # one norm per block
    return norms.sum()
```

The Kronecker layer, written three ways
======

1. Build W, then multiply
------
The direct implementation builds the full $$W$$ and then does an ordinary matrix multiply:

```python
class KronLinear(nn.Module):
    def __init__(self, in_features, out_features, block, rank):
        super().__init__()
        p = in_features // block   # number of block rows
        q = out_features // block  # number of block columns
        self.S = nn.Parameter(torch.ones(p, q))                   # sparsity mask (L1 penalty)
        self.A = nn.Parameter(torch.randn(rank, p, q) / p ** 0.5)
        self.B = nn.Parameter(torch.randn(rank, block, block) / block ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def weight(self):
        W = 0
        for i in range(self.A.shape[0]):
            W = W + torch.kron(self.S * self.A[i], self.B[i])
        return W

    def forward(self, x):
        return x @ self.weight() + self.bias
```

This trains few parameters, but the main multiply `x @ W` costs exactly as much as a dense layer.

2. The vec trick
------
We never need $$W$$ itself, because of the identity

$$(A \otimes B)\,\mathrm{vec}(X) = \mathrm{vec}(B X A^\top).$$

PyTorch uses row vectors (`y = x @ W`) and reshapes row by row, so the identity takes this form: reshape each input row $$x$$ into a $$p \times m$$ matrix $$X$$, and then

$$x\,(A \otimes B) = \mathrm{vec}(A^\top X B).$$

```python
class KronLinearVec(KronLinear):
    def forward(self, x):
        N = x.shape[0]
        p, q = self.S.shape
        m, n = self.B.shape[1:]
        X = x.view(N, p, m)                        # each row becomes a p x m matrix
        Y = 0
        for i in range(self.A.shape[0]):
            A = self.S * self.A[i]
            Y = Y + A.T @ (X @ self.B[i])          # (N, q, n)
        return Y.reshape(N, q * n) + self.bias
```

Per layer, this needs about $$r/m$$ of a dense layer's multiply-adds. So it only saves computation when the rank is smaller than the block size.

3. The vec trick without the loop
------
The loop over $$r$$ launches separate GPU kernels for each term. The whole sum fits in two matrix multiplies. Place the $$B_i$$ side by side to apply all of them at once. Then stack the $$S \odot A_i$$ on top of each other, so that the sum over $$i$$ becomes part of the second multiply's inner dimension:

```python
class KronLinearVecParallel(KronLinear):
    def forward(self, x):
        N = x.shape[0]
        r, p, q = self.A.shape
        m, n = self.B.shape[1:]
        X = x.view(N, p, m)
        B_all = self.B.permute(1, 0, 2).reshape(m, r * n)     # [B_1 | B_2 | ... | B_r]
        Z = X @ B_all                                          # all X @ B_i at once
        Z = Z.view(N, p, r, n).permute(0, 3, 2, 1).reshape(N * n, r * p)
        A_stack = (self.S * self.A).reshape(r * p, q)          # [S*A_1; ...; S*A_r]
        Y = (Z @ A_stack).view(N, n, q)                        # products and sum over r
        return Y.transpose(1, 2).reshape(N, q * n) + self.bias
```

All three versions give the same output, up to float32 rounding (largest difference about $$3 \times 10^{-6}$$ in our runs). Each one also runs under `torch.compile`, which merges small operations into fewer GPU kernels.

Results at batch 256
======
Wall time per training step in ms. Lower is better.

| Baseline | Block | Plain PyTorch | Compiled |
|---|---|---|---|
| Dense | – | **3.70** | 5.00 |
| Dense + group LASSO | 2 × 2 | 5.69 | 5.49 |
| | 4 × 4 | 5.46 | 5.37 |
| | 8 × 8 | 5.40 | 5.37 |

| Kron: rank, block | FLOPs vs dense | Build W | Build W, compiled | Vec loop | Vec loop, compiled | Vec parallel | Vec parallel, compiled |
|---|---|---|---|---|---|---|---|
| r=1, 2 × 2 | 0.50 | 6.60 | 5.87 | 7.68 | 6.12 | 7.34 | 6.18 |
| r=2, 2 × 2 | 1.00 | 8.03 | 6.17 | 10.16 | 6.93 | 7.43 | 6.51 |
| r=4, 2 × 2 | 2.01 | 10.96 | 6.26 | 14.92 | 8.48 | 7.47 | 6.47 |
| r=1, 4 × 4 | 0.25 | 6.49 | 5.62 | 7.52 | 6.08 | 7.14 | 6.01 |
| r=2, 4 × 4 | 0.51 | 7.96 | 6.01 | 10.23 | 7.02 | 7.39 | 6.44 |
| r=4, 4 × 4 | 1.02 | 10.73 | 6.28 | 15.13 | 8.73 | 7.27 | 6.34 |
| r=1, 8 × 8 | 0.13 | 6.56 | 5.76 | 7.64 | 6.21 | 7.33 | 6.12 |
| r=2, 8 × 8 | 0.27 | 8.02 | 6.04 | 10.25 | 7.19 | 7.46 | 6.51 |
| r=4, 8 × 8 | 0.53 | 10.75 | 5.50 | 13.96 | 8.80 | 7.37 | 6.40 |

Peak GPU memory in MB. Lower is better.

| Baseline | Block | Plain PyTorch | Compiled |
|---|---|---|---|
| Dense | – | 57.5 | 57.5 |
| Dense + group LASSO | 2 × 2 | 60.4 | 57.5 |
| | 4 × 4 | 59.7 | 57.5 |
| | 8 × 8 | 59.5 | 57.5 |

| Kron: rank, block | Build W | Build W, compiled | Vec loop | Vec loop, compiled | Vec parallel | Vec parallel, compiled |
|---|---|---|---|---|---|---|
| r=1, 2 × 2 | 48.5 | 40.5 | 41.5 | 40.7 | 41.5 | 40.7 |
| r=2, 2 × 2 | 58.5 | 48.5 | 54.5 | 53.7 | 53.5 | 52.7 |
| r=4, 2 × 2 | 78.5 | 67.5 | 78.5 | 79.7 | 77.5 | 76.7 |
| r=1, 4 × 4 | 35.7 | 28.5 | 28.0 | 27.2 | 28.0 | 27.2 |
| r=2, 4 × 4 | 38.0 | 30.5 | 33.5 | 32.7 | 32.5 | 33.0 |
| r=4, 4 × 4 | 42.5 | 34.5 | 42.5 | 43.7 | 42.5 | 42.7 |
| r=1, 8 × 8 | 32.5 | 25.5 | **24.6** | **24.6** | **24.6** | **24.6** |
| r=2, 8 × 8 | 33.1 | 26.0 | 28.2 | 27.5 | 27.2 | 28.3 |
| r=4, 8 × 8 | 34.2 | 27.0 | 33.5 | 34.7 | 34.2 | 35.4 |

**At this batch size, time does not follow FLOPs.** The rank-1, 8 × 8 layer needs only 13% of the dense layer's multiply-adds, but every Kronecker version is slower than plain dense. The reason is *launch overhead*. For every GPU kernel, the CPU spends roughly 5–20 µs in Python and PyTorch before the GPU can start. At batch 256 each kernel finishes in a few microseconds, so the GPU mostly waits for the CPU, and a step costs roughly (number of kernels) × (cost per launch). The Kronecker layers launch more kernels than one `nn.Linear`, so they lose. Two observations support this. The loop versions slow down as the rank grows, because each extra term adds kernels. The parallel vec trick stays near 7.4 ms at every rank, because its kernel count does not depend on $$r$$.

**Memory is where the Kronecker layer wins at small batch.** Parameters, gradients, and Adam's two state buffers dominate memory here. The rank-1, 8 × 8 layer has 33k parameters instead of 1.05M, and peak memory drops from 57.5 MB to 24.6 MB. Small blocks are the exception. With 2 × 2 blocks, $$S$$ and each $$A_i$$ are 512 × 512, so rank 4 has *more* parameters than the dense layer and uses more memory.

Results at batch 20,000
======
Wall time per training step in ms. Lower is better.

| Baseline | Block | Plain PyTorch | Compiled |
|---|---|---|---|
| Dense | – | 12.13 | 12.25 |
| Dense + group LASSO | 2 × 2 | 12.39 | 12.38 |
| | 4 × 4 | 12.46 | 12.48 |
| | 8 × 8 | 12.54 | 12.56 |

| Kron: rank, block | FLOPs vs dense | Build W | Build W, compiled | Vec loop | Vec loop, compiled | Vec parallel | Vec parallel, compiled |
|---|---|---|---|---|---|---|---|
| r=1, 2 × 2 | 0.50 | 12.40 | 11.74 | 17.81 | 15.92 | 17.33 | 16.12 |
| r=2, 2 × 2 | 1.00 | 12.80 | 11.92 | 33.68 | 29.99 | 23.74 | 22.51 |
| r=4, 2 × 2 | 2.01 | 13.89 | 12.01 | 64.92 | 57.96 | 37.59 | 35.70 |
| r=1, 4 × 4 | 0.25 | 12.42 | 11.74 | 11.99 | 10.12 | 11.44 | 10.20 |
| r=2, 4 × 4 | 0.51 | 12.70 | 11.86 | 21.83 | 18.28 | 15.95 | 14.64 |
| r=4, 4 × 4 | 1.02 | 13.63 | 11.79 | 41.58 | 34.54 | 24.03 | 22.83 |
| r=1, 8 × 8 | 0.13 | 12.41 | 11.75 | 9.59 | 7.52 | 8.94 | **7.49** |
| r=2, 8 × 8 | 0.27 | 12.60 | 11.76 | 16.81 | 12.91 | 11.80 | 10.42 |
| r=4, 8 × 8 | 0.53 | 13.59 | 11.90 | 31.48 | 23.56 | 19.02 | 17.32 |

Peak GPU memory in MB. Lower is better.

| Baseline | Block | Plain PyTorch | Compiled |
|---|---|---|---|
| Dense | – | 432.0 | 359.0 |
| Dense + group LASSO | 2 × 2 | 440.0 | 364.0 |
| | 4 × 4 | 440.0 | 363.3 |
| | 8 × 8 | 440.0 | 363.1 |

| Kron: rank, block | Build W | Build W, compiled | Vec loop | Vec loop, compiled | Vec parallel | Vec parallel, compiled |
|---|---|---|---|---|---|---|
| r=1, 2 × 2 | 428.0 | 351.0 | 581.5 | 598.9 | 581.5 | 599.4 |
| r=2, 2 × 2 | 436.0 | 357.0 | 825.9 | 767.7 | 747.7 | 844.8 |
| r=4, 2 × 2 | 452.0 | 369.0 | 1158.4 | 1256.0 | 1233.2 | 1331.0 |
| r=1, 4 × 4 | 416.0 | 342.0 | 568.2 | 588.9 | 568.2 | 588.9 |
| r=2, 4 × 4 | 418.0 | 343.5 | 804.9 | 747.1 | 727.0 | 826.0 |
| r=4, 4 × 4 | 422.0 | 346.5 | 1122.4 | 1220.5 | 1200.2 | 1299.3 |
| r=1, 8 × 8 | 413.0 | **339.8** | 565.2 | 586.3 | 565.2 | 586.3 |
| r=2, 8 × 8 | 413.5 | 340.1 | 799.6 | 743.0 | 722.1 | 821.3 |
| r=4, 8 × 8 | 414.5 | 340.9 | 1113.4 | 1211.5 | 1192.0 | 1291.2 |

**At a realistic batch size, the vec trick is faster than dense, but only at low rank.** Rank 1 with 8 × 8 blocks takes 7.49 ms per step against 12.13 ms for dense, a 1.6× speed-up. Rank 1 with 4 × 4 blocks and rank 2 with 8 × 8 blocks are also faster than dense. Every other vec-trick setting is slower than dense.

**The speed-up is smaller than the FLOP savings.** Rank 1, 8 × 8 needs 13% of the dense multiply-adds but takes 62% of the time, for two reasons. First, the rest of the step (output layer, ReLU, loss, Adam) does not change. Second, the vec trick writes an intermediate tensor $$Z$$ that is $$r$$ times the size of the input, copies it in the `permute`, and reads it back. For $$r \ge 2$$, this data movement outweighs the FLOP savings.

**Building W gives dense speed with the least memory.** It still runs a dense `x @ W`, so it is never much faster than dense. It has the lowest peak memory, however (340 MB against 359 MB, compiled), because it stores fewer parameters and less optimizer state. The vec trick, in contrast, keeps $$Z$$ for the backward pass. At this batch size, $$Z$$ costs more memory than the smaller parameter count saves, which is why the vec trick uses 1.6–3.7× the memory of dense.

**Group LASSO is cheap at this size, but it is not efficient.** It adds only about 0.3 ms per step. It never makes training faster or lighter, though, because the weights stay dense matrices throughout training.

Takeaways
======
* **Fewer FLOPs do not automatically mean a faster step.** At small batch sizes, the number of kernels matters more than the arithmetic. `torch.compile` helps the Kronecker layers because it merges their many small operations.
* **Choose the rank and the block size together.** The vec trick saves computation only when $$r < m$$. Blocks of 2 × 2 are a poor choice in every setting.
* **Building W is a safe default for memory.** With blocks of 4 × 4 or larger, it uses less memory than dense at both batch sizes, and it is about as fast as dense at large batch.
* **The vec trick is the path to real speed-ups.** It is held back by memory traffic, not by arithmetic.

Next, we plan to write a fused Triton kernel that computes $$Z$$ tile by tile in on-chip memory and never writes it to GPU memory, recomputing it in the backward pass. This should remove both the extra memory and most of the data movement. We also plan to rerun the comparison in mixed precision, where both dense and Kronecker layers use tensor cores.

Run it yourself
======
```bash
git clone https://github.com/KhaliliMahdi/kronvit.git
cd kronvit
python benchmarks/kron_simple_benchmark.py
```

The settings are at the top of the file: `ranks`, `blocks`, `sweep_batch_size`, `steps`, and `repeats`. `run_batch_sweep()` sweeps the batch size at a fixed rank and block size. Each configuration above was run once. In earlier runs with five repeats, the run-to-run standard deviation was usually below 0.1 ms, small enough that a single run is reliable for these comparisons.
