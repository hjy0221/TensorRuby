$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'rubygrad'

nn = Rubygrad::NN
random = Random.new(42)
x = Rubygrad.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
target = Rubygrad.tensor([[0], [1], [1], [0]])

model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Linear.new(8, 1, random: random)
)
criterion = nn::MSELoss.new
optimizer = Rubygrad::Optim::SGD.new(model.parameters, lr: 0.1)

500.times do
  optimizer.zero_grad
  loss = criterion.call(model.call(x), target)
  loss.backward
  optimizer.step
end

output = model.call(x)
predictions = output.to_a.flatten.map { |score| score >= 0.5 ? 1 : 0 }
puts format('Final loss: %.8g', criterion.call(output, target).data)
puts "Predictions: #{predictions.inspect}"
abort 'XOR training failed' unless predictions == [0, 1, 1, 0]
