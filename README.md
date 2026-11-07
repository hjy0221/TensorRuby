# TensorRuby

[English](README.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

TensorRuby is a small deep learning framework written in pure Ruby. It requires Ruby 2.6 or later and has no runtime dependencies.

[Development roadmap](ROADMAP.md)

## Installation

Install the gem from the latest GitHub release.

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
```

## Train your own model

After installing the gem, list and run the bundled examples from any directory. For your own Ruby program, use `require "tensorruby"`. A trainer consumes `[input, target]` pairs, and each pair can contain a full tensor batch.

```sh
tensorruby examples
tensorruby run xor
```

```ruby
require 'tensorruby'

model = TensorRuby::NN::Linear.new(1, 1)
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new
)
data = [[TensorRuby.tensor([[0], [1], [2]]), TensorRuby.tensor([[1], [3], [5]])]]
200.times { trainer.train_epoch(data) }
p model.call([[3]]).to_a
```

## Getting started

```sh
ruby -Ilib test/tensorruby_test.rb
ruby bin/tensorruby run regression
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

Create tensors with `TensorRuby.tensor` and compose layers with `NN::Sequential`. Run the complete example with `tensorruby run xor`.

## Multiclass classification

Use `NN::CrossEntropyLoss` with raw logits shaped `[batch, classes]` and integer class indices shaped `[batch]`. `Optim::Adam` keeps its running moments across training steps. Run the three-class example with `tensorruby run classification`.

`Data::DataLoader.new(samples, batch_size: 8)` groups `[input, target]` pairs into tensor mini-batches and shuffles them each epoch.

## Training and evaluation

Use `model.train` to enable dropout and `model.eval` to disable it in the model and its nested layers. Wrap inference in `TensorRuby.no_grad` to skip recording new operations for backpropagation.

```ruby
model.eval
predictions = TensorRuby.no_grad { model.call(x).to_a }
```

## GPT and diffusion examples

The GPT-2-style decoder accepts integer token IDs and trains with next-token cross entropy. It includes learned token and position embeddings, causal multi-head attention, pre-layer normalization, GELU, and a tied output projection. The diffusion examples train small noise predictors on synthetic points and grayscale stripe patterns. The image example writes a PNG contact sheet of eight generated 6×6 samples.

```sh
tensorruby run gpt
tensorruby run transformer
tensorruby run diffusion
tensorruby run image_diffusion diffusion-samples.png diffusion-trajectory.png
```

## Chat and monster sprites

Train a small chat model on the bundled Korean, Japanese, and English dialogues, or pass a JSON array of dialogues containing `[user, assistant]` pairs via `--data FILE`. The demo uses a closed whitespace vocabulary. The sprite demo uses fixed silhouettes and diffusion-generated color and eye variations.

```sh
tensorruby run chat --interactive
tensorruby run monster_diffusion all monster-sprites.png
```

## Train and reload a multilingual capital model

Train from scratch on 12 country–capital facts in three languages, then reload the resulting weights without retraining. The evaluation reports exact match for trained questions and reordered questions using the same words.

```sh
tensorruby run capital_chat --save capital-chat-weights.json
tensorruby run capital_chat --load capital-chat-weights.json --interactive
```

## Pretrain on raw text

Train a byte-pair tokenizer and a small GPT from scratch. The tokenizer can encode any valid UTF-8 text. Pass separate UTF-8 training and validation files with one document per line. The tokenizer learns merges from training text only.

```sh
tensorruby run pretrain
tensorruby run pretrain --train train.txt --validation validation.txt --epochs 12
```

## Fine-tune a checkpoint and generate text

Pretraining accumulates token-weighted gradients across sequences, clips their global norm, and uses warmup followed by cosine decay. The JSON checkpoint stores model weights, configuration, and byte-pair merges. Fine-tuning uses assistant tokens only.

```sh
tensorruby run pretrain --save pretrained.json
tensorruby run finetune --save chat-model.json pretrained.json
tensorruby run generate --chat chat-model.json "하늘?"
```

## Backend roadmap

Each tensor operation records one node with a vector–Jacobian product. Use `output.backward(seed)` with a seed of the same shape, or omit the seed for a single-element output. The CPU kernels currently run in Ruby. A future backend can replace those kernels with native implementations.
