# Quickstart

[Home](../../README.md) · [Quickstart](quickstart.md) · [API reference](api.md) · [Examples](examples.md) · [Development](development.md)

[English](../en/quickstart.md) · [日本語](../ja/quickstart.md) · [한국어](../ko/quickstart.md)

Run the following commands in a terminal. The gem includes the examples, so they run from any directory.

```sh
curl -fL https://github.com/hjy0221/TensorRuby/releases/latest/download/tensorruby.gem -o tensorruby.gem
gem install --local tensorruby.gem
tensorruby run xor
```

## Train a model

Save this as `train.rb` and run `ruby train.rb`. The model learns `y = 2x + 1`. Each pair passed to the trainer contains one input batch and its targets.

```ruby
require 'tensorruby'

x = TensorRuby.tensor([[0], [1], [2]])
y = TensorRuby.tensor([[1], [3], [5]])
model = TensorRuby::NN::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::MSELoss.new
)

data = [[x, y]]
200.times { trainer.train_epoch(data) }
p trainer.evaluate(data)
model.eval
p TensorRuby.no_grad { model.call([[3]]).to_a } # approximately [[7.0]]
```

## Next steps

Use `tensorruby examples` to list the available examples. If the shell cannot find the command, add the gem executable directory to your PATH. Run `gem environment` to locate it.
