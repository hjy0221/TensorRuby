$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(42)
centers = [[-2, -1], [2, -1], [0, 2]]
sample = lambda do |count|
  labels = centers.each_index.flat_map { |label| Array.new(count, label) }
  features = labels.map { |label| centers[label].map { |coordinate| coordinate + random.rand - 0.5 } }
  [TensorRuby.tensor(features), labels]
end

x, labels = sample.call(8)
test_x, test_labels = sample.call(5)
model = nn::Sequential.new(
  nn::Linear.new(2, 8, random: random),
  nn::ReLU.new,
  nn::Dropout.new(0.15, random: random),
  nn::Linear.new(8, 3, random: random)
)
loss_fn = nn::CrossEntropyLoss.new
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.03)
target = TensorRuby.tensor(labels)

trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: loss_fn, weight: ->(targets) { targets.size }
)
training_data = [[x, target]]
initial_loss = trainer.evaluate(training_data)
loader = TensorRuby::Data::DataLoader.new(x.to_a.zip(labels), batch_size: 8, random: random)
40.times { trainer.train_epoch(loader) }

model.eval
final_loss, predictions = TensorRuby.no_grad do
  loss = loss_fn.call(model.call(x), target).data
  classes = model.call(test_x).to_a.map { |scores| scores.each_index.max_by { |i| scores[i] } }
  [loss, classes]
end
correct = predictions.zip(test_labels).count { |prediction, label| prediction == label }
puts format('Training loss: %.6f -> %.6f', initial_loss, final_loss)
puts "Held-out accuracy: #{correct}/#{test_labels.size}"
abort 'Classification training failed' unless final_loss < initial_loss && correct == test_labels.size
