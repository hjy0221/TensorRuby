$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'optparse'

options = { epochs: 12 }
OptionParser.new do |parser|
  parser.on('--train FILE') { |file| options[:train] = file }
  parser.on('--validation FILE') { |file| options[:validation] = file }
  parser.on('--epochs N', Integer) { |count| options[:epochs] = count }
end.parse!
abort 'epochs must be positive' unless options[:epochs].positive?
if options.key?(:train) != options.key?(:validation)
  abort 'provide both --train and --validation for a custom corpus'
end

train = [
  '오늘은 맑아요. 내일도 맑아요.',
  '오늘은 비가 와요. 내일은 맑아요.',
  '今日は晴れです。明日も晴れです。',
  '今日は雨です。明日は晴れです。',
  'Today is sunny. Tomorrow is sunny.',
  'Today is rainy. Tomorrow is sunny.'
]
validation = [
  '오늘은 맑아요. 내일은 비가 와요.',
  '今日は晴れです。明日は雨です。',
  'Today is sunny. Tomorrow is rainy.'
]
train = File.readlines(options[:train], encoding: 'UTF-8', chomp: true) if options[:train]
validation = File.readlines(options[:validation], encoding: 'UTF-8', chomp: true) if options[:validation]
tokenizer = TensorRuby::Text::BytePairTokenizer.train(train, max_merges: 48)
context = 16
windows = TensorRuby::Text::NextTokenWindows
train_pairs = windows.new(train, tokenizer: tokenizer, context_size: context).to_a
validation_pairs = windows.new(validation, tokenizer: tokenizer, context_size: context).to_a
abort 'training and validation both need at least one next-token pair' if train_pairs.empty? || validation_pairs.empty?

random = Random.new(42)
model = TensorRuby::NN::GPT.new(vocab_size: tokenizer.size, context_size: context, features: 8,
                               heads: 2, layers: 1, random: random)
criterion = TensorRuby::NN::CrossEntropyLoss.new
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.015)
validation_loss = lambda do
  TensorRuby.no_grad do
    total = validation_pairs.sum do |input, targets|
      criterion.call(model.call(input), targets).data * targets.size
    end
    total / validation_pairs.sum { |_, targets| targets.size }
  end
end

model.eval
before = validation_loss.call
model.train
options[:epochs].times do
  train_pairs.shuffle(random: random).each do |input, targets|
    optimizer.zero_grad
    criterion.call(model.call(input), targets).backward
    optimizer.step
  end
end
model.eval
after = validation_loss.call
puts "Vocabulary: #{tokenizer.size} tokens (#{tokenizer.merges.size} learned merges)"
puts format('Held-out next-token loss: %.4f -> %.4f', before, after)
abort 'Training produced nonfinite loss' unless after.finite?
abort 'Validation loss did not improve' if !options[:train] && !options[:validation] && after >= before
