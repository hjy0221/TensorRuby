# API reference

[Home](../../README.md) · [Quickstart](quickstart.md) · [API reference](api.md) · [Examples](examples.md) · [Development](development.md)

[English](../en/api.md) · [日本語](../ja/api.md) · [한국어](../ko/api.md)

Signatures use the `TensorRuby` namespace unless shown otherwise.

## Tensor and autograd

[Tensor and autograd](tensor.md) — Shape rules, operation signatures, seeds, and gradient accumulation.

## Neural networks

- `NN::Module#parameters` collects trainable scalar values. `train` and `eval` propagate the mode to nested layers.
- `NN::Sequential.new(*layers)` applies layers in order. `NN::Linear.new(in_features, out_features)` accepts rank-2 or higher input with features on the last axis.
- `NN::ReLU`, `NN::GELU`, `NN::LayerNorm.new(features)`, and `NN::Dropout.new(p = 0.5)` provide activations, normalization, and dropout. LayerNorm takes a matrix. Dropout runs in training mode.
- `NN::MSELoss#call(prediction, target)` requires equal shapes. `NN::CrossEntropyLoss.new(ignore_index: nil)` takes raw `[batch, classes]` logits and `[batch]` class indices.

## Training and data

- `Data::DataLoader.new(dataset, batch_size:, shuffle: true, random: Random.new)` yields tensor pairs from an indexable dataset. The final partial batch is included.
- `Optim::SGD.new(parameters, lr:)` and `Optim::Adam.new(parameters, lr: 0.001)` support `zero_grad` and `step`. Adam also accepts `step(lr:)`.
- [Trainer and mini-batches](trainer.md) — Weighted losses, partial batches, gradient accumulation, and evaluation.

## Text and generation

- `NN::SelfAttention` and `NN::TransformerBlock` operate on one `[sequence, features]` matrix. `NN::GPT.new(vocab_size:, context_size:, features:, heads:, layers:)` accepts one integer token sequence and returns `[sequence, vocabulary]` logits.
- `GPT#generate(ids, max_new_tokens:, stop_token: nil)` performs greedy decoding and returns the prompt plus generated IDs.
- `Text::BytePairTokenizer.train(documents, max_merges: 64)` learns UTF-8 byte merges. `Text::NextTokenWindows` yields input and shifted target IDs.
- `Text::Checkpoint.save(path, model, vocabulary)` stores configuration, byte merges, and weights. `load(path)` returns a model and byte vocabulary. `Chat::Session#reply` uses that vocabulary to encode dialogue and returns decoded text.

## Diffusion

- `Diffusion::Schedule.new(steps:, beta_start: 0.0001, beta_end: 0.02)` defines a linear DDPM schedule.
- `add_noise(clean, noise, step)` uses a zero-based step. `sample(shape, on_step: nil) { |noisy, step| … }` expects a same-shape noise prediction. The optional observer receives the number of remaining steps and the current sample.
