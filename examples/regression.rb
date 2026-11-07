$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

# Fit y = 2x + 1 using one batch of five observations.
nn = TensorRuby::NN
inputs = TensorRuby.tensor([[-2], [-1], [0], [1], [2]])
targets = TensorRuby.tensor([[-3], [-1], [1], [3], [5]])
model = nn::Linear.new(1, 1, random: Random.new(42))
optimizer = TensorRuby::Optim::SGD.new(model.parameters, lr: 0.05)
trainer = TensorRuby::Training::Trainer.new(model, optimizer: optimizer, loss: nn::MSELoss.new)

data = [[inputs, targets]]
initial = trainer.evaluate(data)
200.times { trainer.train_epoch(data) }
final = trainer.evaluate(data)
puts format('Regression MSE: %.6f → %.8g', initial, final)
puts format('Learned equation: y = %.6fx + %.6f', model.weight[0, 0].data, model.bias[0].data)
abort 'Regression did not converge' unless final < 1e-8
