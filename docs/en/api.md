# API reference

[Home](../../README.md) · [Quickstart](quickstart.md) · [API reference](api.md) · [Examples](examples.md) · [Development](development.md)

[English](../en/api.md) · [日本語](../ja/api.md) · [한국어](../ko/api.md)

Signatures use the `TensorRuby` namespace unless shown otherwise.

## Tensors and autograd

`TensorRuby.tensor(data, shape: nil)`

- Create a tensor from numeric data or nested arrays. Supply `shape:` to interpret flat data with an explicit layout.
- `shape`, `strides`, `rank`, and `size` describe the tensor. `to_a` reads numeric values. `tensor[i, j]` and `values` expose scalar `Value` objects.
- Elementwise `+`, `-`, `*`, and `/` broadcast trailing axes. `matmul(other)` multiplies the last two axes of rank-2 or higher inputs and broadcasts leading axes. `[2, 3, 4] × [4, 5]` yields `[2, 3, 5]`.
- `sum` and `mean` reduce the entire tensor to a `Value`. `transpose`, `softmax(causal: false)`, `normalize_rows(eps: 1e-5)`, `gather_rows(indices)`, and `concat_columns(other)` operate on matrices.
- `Value#backward(seed = 1.0)` accumulates leaf gradients. `Tensor#backward(seed)` accepts a seed of the same shape, and allows it to be omitted for one element. `TensorRuby.no_grad { … }` disables recording new operations.

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[1, 2], [3, 4]])
w = TensorRuby.tensor([[2], [1]])
loss = x.matmul(w).sum
loss.backward
p w.values.map(&:grad) # [4.0, 6.0]
```

## Neural networks

- `NN::Module#parameters` collects trainable scalar values. `train` and `eval` propagate the mode to nested layers.
- `NN::Sequential.new(*layers)` applies layers in order. `NN::Linear.new(in_features, out_features)` accepts rank-2 or higher input with features on the last axis.
- `NN::ReLU`, `NN::GELU`, `NN::LayerNorm.new(features)`, and `NN::Dropout.new(p = 0.5)` provide activations, normalization, and dropout. LayerNorm takes a matrix. Dropout runs in training mode.
- `NN::MSELoss#call(prediction, target)` requires equal shapes. `NN::CrossEntropyLoss.new(ignore_index: nil)` takes raw `[batch, classes]` logits and `[batch]` class indices.

## Training and data

- `Data::DataLoader.new(dataset, batch_size:, shuffle: true, random: Random.new)` yields tensor pairs from an indexable dataset. The final partial batch is included.
- `Optim::SGD.new(parameters, lr:)` and `Optim::Adam.new(parameters, lr: 0.001)` support `zero_grad` and `step`. Adam also accepts `step(lr:)`.
- `Training::Trainer.new(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)` consumes `[input, target]` pairs. `train_epoch(examples, accumulation_steps: 1, learning_rate: nil)` returns mean loss, and `evaluate(examples)` evaluates without recording gradients. A learning-rate callable receives the optimizer step, starting at zero.

## Text and generation

- `NN::SelfAttention` and `NN::TransformerBlock` operate on one `[sequence, features]` matrix. `NN::GPT.new(vocab_size:, context_size:, features:, heads:, layers:)` accepts one integer token sequence and returns `[sequence, vocabulary]` logits.
- `GPT#generate(ids, max_new_tokens:, stop_token: nil)` performs greedy decoding and returns the prompt plus generated IDs.
- `Text::BytePairTokenizer.train(documents, max_merges: 64)` learns UTF-8 byte merges. `Text::NextTokenWindows` yields input and shifted target IDs.
- `Text::Checkpoint.save(path, model, vocabulary)` stores configuration, byte merges, and weights. `load(path)` returns a model and byte vocabulary. `Chat::Session#reply` uses that vocabulary to encode dialogue and returns decoded text.

## Diffusion

- `Diffusion::Schedule.new(steps:, beta_start: 0.0001, beta_end: 0.02)` defines a linear DDPM schedule.
- `add_noise(clean, noise, step)` uses a zero-based step. `sample(shape, on_step: nil) { |noisy, step| … }` expects a same-shape noise prediction. The optional observer receives the number of remaining steps and the current sample.
