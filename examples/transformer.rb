$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(17)
model = nn::Sequential.new(
  nn::TransformerBlock.new(5, hidden_features: 12, random: random),
  nn::Linear.new(5, 2, random: random)
)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.02)
loss_fn = nn::CrossEntropyLoss.new

# Recall the first bit at every position, including after distractor bits.
samples = [0, 1].repeated_permutation(3).map do |bits|
  features = bits.each_with_index.map do |bit, position|
    [1 - bit, bit] + Array.new(3) { |i| i == position ? 1 : 0 }
  end
  [TensorRuby.tensor(features), TensorRuby.tensor([bits.first] * bits.size)]
end

trainer = TensorRuby::Training::Trainer.new(model, optimizer: optimizer, loss: loss_fn)
60.times { trainer.train_epoch(samples.shuffle(random: random)) }

model.eval
correct = TensorRuby.no_grad do
  samples.sum do |input, labels|
    predictions = model.call(input).to_a.map { |scores| scores.each_index.max_by { |i| scores[i] } }
    predictions.zip(labels.to_a).count { |prediction, label| prediction == label }
  end
end
total = samples.sum { |_, labels| labels.size }
puts "First-bit recall: #{correct}/#{total} training positions"
abort 'Transformer training failed' unless correct == total
