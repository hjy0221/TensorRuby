$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(42)
inputs = TensorRuby.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = [0, 1, 1, 0]
data = [[inputs, labels]]
model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Linear.new(8, 2, random: random)
)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.04)
trainer = TensorRuby::Training::Trainer.new(model, optimizer: optimizer, loss: nn::CrossEntropyLoss.new)

before = trainer.evaluate(data)
200.times { trainer.train_epoch(data) }
after = trainer.evaluate(data)
model.eval
predictions = TensorRuby.no_grad do
  model.call(inputs).to_a.map { |scores| scores.each_index.max_by { |i| scores[i] } }
end
puts format('XOR cross entropy: %.6f → %.6f', before, after)
puts "XOR predictions: #{predictions.inspect}"
abort 'XOR training failed' unless predictions == labels
