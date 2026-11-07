# TensorRuby roadmap

[English](ROADMAP.md) · [日本語](ROADMAP.ja.md) · [한국어](ROADMAP.ko.md)

The long-term target is a general-purpose Ruby framework capable of running practical PyTorch-style research and training workloads. Progress is measured in supported workloads, numerical agreement, performance, and ease of installation. API names follow Ruby conventions.

| Milestone | Deliverables | Exit criteria |
| --- | --- | --- |
| 1. Release and measurement baseline | A Ruby compatibility matrix, installed-gem checks, a repeatable matrix multiply benchmark, and an isolated CPU kernel boundary | CI runs tests and an installed example on Ruby 2.6, 3.1, and 3.4. Forward and backward matrix multiplication can be benchmarked with the same command on two revisions. |
| 2. Tensor runtime | Separate storage from shape, stride, and offset metadata. Add dtype rules, reshape and transpose views, axis reductions, and batched matrix multiplication. | Cover scalar, noncontiguous, broadcast, and batched cases with reference fixtures and gradient checks. Preserve existing training examples. |
| 3. Training at realistic sizes | Add named parameter state, optimizer checkpoints, deterministic training resume, and reproducible data splits. Extend CI to cover these workflows. | An interrupted deterministic run reproduces the weights and losses of an uninterrupted run. A fresh install can reproduce a complete training run from documented commands. |
| 4. Accelerated CPU and device backends | Keep Ruby as a reference backend. Add native CPU kernels, then device storage and dispatch. Start GPU support on one backend before expanding. | Each backend runs the same operator and gradient checks. Publish timing and memory by shape and dtype for matrix multiplication, a Transformer block, and an image training step. Target native CPU matrix multiplication within 2× of PyTorch CPU on matched hardware and thread counts for a documented shape suite. |
| 5. Model coverage | Add convolution and spatial normalization, batched attention and generation caches, checkpoint import, and reusable vision and text modules. | Train and reload a small CNN on an image dataset, run a compact GPT with a public checkpoint format, and train a small convolutional diffusion model. Report task metrics, throughput, and peak memory. |
| 6. Broader framework coverage | Expand dtype and device coverage, operator APIs, multi-device training, packaging, and user documentation. | Maintain a public compatibility matrix for supported operators and devices, measure distributed scaling and checkpoint recovery, and verify release installation across supported platforms. |

Milestones are dependencies rather than calendar promises. Use fixed seeds and publish environment details with performance results. Keep the Pure Ruby runtime usable throughout the transition.

## Repeatable benchmark

Run from the repository root:

```sh
SIZE=48 ITERATIONS=20 ruby -Ilib benchmark/matmul.rb
```

The benchmark reports elapsed time and Ruby objects allocated per iteration for forward inference and forward plus backward. Compare revisions on the same machine, Ruby version, shape, and iteration count. This benchmark measures one CPU workload; it does not estimate model-level speed or device parity.
