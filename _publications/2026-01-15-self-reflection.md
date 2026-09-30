---
title: "From Emergence to Control: Probing and Modulating Self-Reflection in Language Models"
collection: publications
permalink: /publication/self-reflection
excerpt: 'Self-reflection is already present, though rare, in pretrained LLMs: injecting reasoning traces from a fine-tuned model raises its frequency, hidden states separate reflective from non-reflective contexts, and a single self-reflection vector enhances or suppresses reflection at inference time.'
date: 2026-01-15
venue: 'Transactions on Machine Learning Research (TMLR)'
paperurl: 'https://arxiv.org/abs/2506.12217'
citation: 'X. Zhu, J. Jiang, M. Khalili, Z. Zhu, &quot;From Emergence to Control: Probing and Modulating Self-Reflection in Language Models,&quot; <i>Transactions on Machine Learning Research (TMLR)</i>, 2026.'
---

**Xudong Zhu, Jiachen Jiang, Mahdi Khalili, Zhihui Zhu**

[Paper (arXiv)](https://arxiv.org/abs/2506.12217) &nbsp;·&nbsp; **Code** (coming soon) &nbsp;·&nbsp; [BibTeX](#bibtex)

![Self-reflection frequency of pretrained and fine-tuned models on MATH500](/images/publications/self-reflection-overview.png)
*Left: fraction of MATH500 responses showing self-reflection for the pretrained Qwen2.5-1.5B (A<sub>pt</sub>), for A<sub>pt</sub> with reflection-inducing probing (CoT injected from the fine-tuned model), and for the fine-tuned DeepSeek-R1-Distill-Qwen-1.5B (A<sub>ft</sub>). Right: an example of spontaneous self-reflection in the pretrained model.*

Abstract
======
Self-reflection—the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning—has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, *we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models*. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6% to 18.6%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, *we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning*. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control.

Key findings
======
* **Self-reflection already emerges during pretraining.** On MATH500, the pretrained Qwen2.5-1.5B produces self-reflection in 0.6% of responses, compared with 96.8% for its fine-tuned counterpart DeepSeek-R1-Distill-Qwen-1.5B.
* **Reflection-inducing probing exposes the latent capacity.** Feeding the pretrained model the fine-tuned model's reasoning up to the point where it reflects raises the pretrained model's self-reflection frequency from 0.6% to 18.6%.
* **Hidden states separate reflective from non-reflective contexts.** Comparing the hidden state of the token just before "wait" with the same token in contexts that do not lead to reflection, both the pretrained and fine-tuned models show clearly separated clusters (UMAP, layer 15 of 28); the separation grows with depth.
* **A single direction gives bidirectional control.** The difference of mean hidden states between the two contexts defines a self-reflection vector. Adding it with a positive or negative scale enhances or suppresses reflection at inference time, with no additional training. Vectors extracted from GPQA Diamond transfer to MATH500.

Results
======
![Effect of steering strength on response length and accuracy](/images/publications/self-reflection-steering.png)
*Effect of the steering strength α on Pass@1 and average response length for DeepSeek-R1-Distill-Qwen-1.5B on MATH-500, with the self-reflection vector injected at layer 14. Negative α shortens responses; moderate positive α lengthens them and improves accuracy, which falls again at larger α.*

MATH-500 Pass@1 (%) and average response length in tokens (in parentheses). BF is budget forcing (appending "wait"); SR is self-reflection steering.

| Model | Vanilla | BF | SR Enhanced | SR Suppressed |
|---|---|---|---|---|
| DeepSeek-R1-1.5B | 84.1 (4755) | 85.5 (10122) | **87.4** (9458) | 83.2 (**3716**) |
| DeepSeek-R1-7B | 92.7 (3585) | 93.1 (7111) | **93.5** (8959) | 91.2 (**2439**) |
| Qwen2.5 1.5B | 27.5 (1515) | 29.6 (3993) | **36.9** (1836) | – |
| Qwen2.5 7B | 44.8 (1294) | 46.0 (3986) | **56.8** (2671) | – |
| Llama 3.1 8B Instruct | 44.0 (1613) | 40.1 (3771) | **57.7** (2887) | – |

Enhancement adds 12.0 points on Qwen2.5 7B and 13.7 points on Llama 3.1 8B Instruct. Suppression cuts the response length of DeepSeek-R1-7B by 32% while Pass@1 stays above 91%. The paper also reports AIME 2024 and GPQA Diamond results with the same comparison.

BibTeX
======
{% raw %}
```bibtex
@article{zhu2026reflection,
  title   = {From Emergence to Control: Probing and Modulating Self-Reflection in Language Models},
  author  = {Zhu, Xudong and Jiang, Jiachen and Khalili, Mohammad Mahdi and Zhu, Zhihui},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```
{% endraw %}
