$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'rubygrad'
x = Rubygrad::Tensor.new([[-2], [-1], [0], [1], [2]])
y = Rubygrad::Tensor.new([[-3], [-1], [1], [3], [5]])
model = Rubygrad::NN::Linear.new(1, 1, random: Random.new(42))
loss_fn = Rubygrad::NN::MSELoss.new
optimizer = Rubygrad::Optim::SGD.new(model.parameters, lr: 0.05)
201.times do |epoch|
  optimizer.zero_grad
  loss = loss_fn.call(model.call(x), y)
  loss.backward
  optimizer.step
  puts "epoch #{epoch}: loss=#{format('%.8f', loss.data)}" if (epoch % 50).zero?
end
puts "weight=#{model.weight.to_a.inspect}, bias=#{model.bias.to_a.inspect}"
