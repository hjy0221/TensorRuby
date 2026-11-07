$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'optparse'

options = { epochs: 12, batch_size: 2 }
OptionParser.new do |parser|
  parser.on('--train FILE') { |file| options[:train] = file }
  parser.on('--validation FILE') { |file| options[:validation] = file }
  parser.on('--batch-size N', Integer) { |count| options[:batch_size] = count }
  parser.on('--save FILE') { |file| options[:save] = file }
  parser.on('--epochs N', Integer) { |count| options[:epochs] = count }
end.parse!
abort 'epochs and batch size must be positive' unless options.values_at(:epochs, :batch_size).all?(&:positive?)
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
vocabulary = TensorRuby::Chat::ByteVocabulary.new(tokenizer)
context = 24
windows = TensorRuby::Text::NextTokenWindows
train_pairs = windows.new(train, tokenizer: vocabulary, context_size: context).to_a
validation_pairs = windows.new(validation, tokenizer: vocabulary, context_size: context).to_a
abort 'training and validation both need at least one next-token pair' if train_pairs.empty? || validation_pairs.empty?

random = Random.new(42)
model = TensorRuby::NN::GPT.new(vocab_size: vocabulary.size, context_size: context, features: 8,
                               heads: 2, layers: 1, random: random)
optimizer = TensorRuby::Optim::Adam.new(model.parameters)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::CrossEntropyLoss.new,
  weight: ->(targets) { targets.size }, max_grad_norm: 1.0
)

batch_size = options[:batch_size]
updates = options[:epochs] * ((train_pairs.size + batch_size - 1) / batch_size)
warmup = [updates / 10, 10].min
schedule = lambda do |step|
  if step < warmup
    0.025 * (step + 1).fdiv(warmup)
  else
    progress = (step - warmup).fdiv([updates - warmup - 1, 1].max)
    0.005 + 0.01 * (1 + Math.cos(Math::PI * progress))
  end
end
before = trainer.evaluate(validation_pairs)
options[:epochs].times do
  trainer.train_epoch(train_pairs.shuffle(random: random), accumulation_steps: batch_size, learning_rate: schedule)
end
after = trainer.evaluate(validation_pairs)
puts "Vocabulary: #{vocabulary.size} tokens (#{tokenizer.merges.size} learned merges)"
puts format('Held-out next-token loss: %.4f -> %.4f', before, after)
abort 'Training produced nonfinite loss' unless after.finite?
abort 'Validation loss did not improve' if !options[:train] && !options[:validation] && after >= before
TensorRuby::Text::Checkpoint.save(options[:save], model, vocabulary) if options[:save]
