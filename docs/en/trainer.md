# Trainer and mini-batches

[API reference](api.md) · [Quickstart](quickstart.md) · [Examples](examples.md)

[English](../en/trainer.md) · [日本語](../ja/trainer.md) · [한국어](../ko/trainer.md)

## Configuration

`TensorRuby::Training::Trainer.new(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)`

- `optimizer:` implements `zero_grad` and `step`. `loss:` receives the prediction and target and returns a finite scalar `Value`.
- `weight:` defaults to 1 per pair. Use the target element count for an elementwise mean or the number of supervised tokens for a masked language-model loss. A zero-weight pair is skipped.
- `max_grad_norm:` clips the global gradient norm before each update. `forward:` can supply a custom callable in place of `model.call(input)`.

## Training and evaluation

| API | Input and return value |
| --- | --- |
| `train_epoch(examples, accumulation_steps: 1, learning_rate: nil)` | Sets training mode and returns the weighted mean loss. Each group of up to `accumulation_steps` pairs produces one update. A partial group is kept. |
| `evaluate(examples)` | Evaluates with gradients disabled and returns the weighted mean loss. The model’s previous training or evaluation mode is restored. |
| `steps` | Counts completed optimizer updates, including those in earlier epochs. |

`learning_rate:` receives the zero-based update number and returns the learning rate. The optimizer must accept `step(lr:)`; `Optim::Adam` does.

## Worked example

The loader yields batches of two and one. Weighting by target count gives each observation the same influence, and accumulating both batches makes one optimizer update.

```ruby
require 'tensorruby'

samples = [[[0], [1]], [[1], [3]], [[2], [5]]]
loader = TensorRuby::Data::DataLoader.new(samples, batch_size: 2, shuffle: false)
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new,
  weight: ->(target) { target.size }
)

trainer.train_epoch(loader, accumulation_steps: 2)
p trainer.steps                 # 1
300.times { trainer.train_epoch(loader, accumulation_steps: 2) }
p trainer.evaluate(loader)      # approximately 0.0
```

## Errors

A nonpositive accumulation count, a negative or nonfinite weight, a nonfinite loss, or an epoch with no positively weighted samples raises `ArgumentError`. Training inside `no_grad` also raises `ArgumentError`.
