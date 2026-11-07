# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby is a small deep learning framework written in pure Ruby. It requires Ruby 2.6 or later and has no runtime dependencies.

## Installation

Install the gem from the latest GitHub release.

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
```

## Getting started

```sh
ruby -Ilib test/tensorruby_test.rb
ruby examples/regression.rb
gem build tensorruby.gemspec
```

Run these commands from the project root. Tests require Minitest. If Rake is installed, you can also run `rake test`.

## Usage

After installing, run the snippet with `ruby`.

```ruby
require 'tensorruby'
x = TensorRuby::Tensor.new([[1, 2], [3, 4]])
w = TensorRuby::Tensor.new([[2], [1]])
output = x.matmul(w) + 1
output.sum.backward
p output.to_a            # [[5.0], [11.0]]
p w.values.map(&:grad)  # [4.0, 6.0]
```

`Value#backward(seed = 1.0)` traverses the computation graph in reverse topological order and sums gradient contributions at shared nodes. Each call clears intermediate gradients and accumulates leaf gradients. Call `optimizer.zero_grad` before each training step, and build a fresh forward graph after updating parameters.

`Tensor` uses contiguous row-major storage with shape and stride metadata. Indexing and `values` expose scalar `Value` objects. Elementwise `+`, `-`, `*`, and `/` broadcast across trailing axes. `matmul` accepts rank-2 matrices. `sum` and `mean` reduce all elements to a `Value`. `Linear` expects input shaped `[batch, features]`, and MSE requires identical prediction and target shapes. The current API supports nonempty dimensions and whole-tensor reductions.

## Train an XOR model

Create tensors with `TensorRuby.tensor` and compose layers with `NN::Sequential`. Run the complete example with `ruby examples/xor.rb`.

```ruby
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(42)
x = TensorRuby.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
target = TensorRuby.tensor([[0], [1], [1], [0]])

model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Linear.new(8, 1, random: random)
)
criterion = nn::MSELoss.new
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.1)

500.times do
  optimizer.zero_grad
  loss = criterion.call(model.call(x), target)
  loss.backward
  optimizer.step
end
```

## Multiclass classification

Use `NN::CrossEntropyLoss` with raw logits shaped `[batch, classes]` and integer class indices shaped `[batch]`. `Optim::Adam` keeps its running moments across training steps. Run the three-class example with `ruby examples/classification.rb`.

## Backend roadmap

Each tensor operation records one node with a vector–Jacobian product. Use `output.backward(seed)` with a seed of the same shape, or omit the seed for a single-element output. The CPU kernels currently run in Ruby. A future backend can replace those kernels with native implementations.
