$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'optparse'

options = { epochs: 40 }
OptionParser.new do |parser|
  parser.banner = 'Usage: ruby examples/finetune.rb [--data dialogues.json] [--save FILE] PRETRAINED.json'
  parser.on('--data FILE') { |path| options[:data] = path }
  parser.on('--save FILE') { |path| options[:save] = path }
  parser.on('--epochs N', Integer) { |count| options[:epochs] = count }
end.parse!
abort 'provide one pretrained checkpoint and a positive epoch count' unless ARGV.size == 1 && options[:epochs].positive?

model, vocabulary = TensorRuby::Text::Checkpoint.load(ARGV.first)
dialogues = [
  [['하늘?', '맑아요.']],
  [['空？', '晴れです。']],
  [['Sky?', 'Sunny.']]
]
dialogues = JSON.parse(File.read(options[:data])) if options[:data]
examples = dialogues.map { |turns| vocabulary.training_example(turns) }
context = model.config.fetch(:context_size)
if examples.empty? || examples.any? { |input, _| input.size > context }
  abort "provide nonempty dialogues that fit the #{context}-token context"
end

ignored = TensorRuby::Chat::Vocabulary::IGNORE
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.015)
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: TensorRuby::NN::CrossEntropyLoss.new(ignore_index: ignored),
  weight: ->(targets) { targets.count { |token| token != ignored } }
)
random = Random.new(7)
before = trainer.evaluate(examples)
options[:epochs].times { trainer.train_epoch(examples.shuffle(random: random)) }
after = trainer.evaluate(examples)
model.eval
puts format('Assistant-token training loss: %.4f -> %.4f', before, after)
abort 'Training produced nonfinite loss' unless after.finite?
session = TensorRuby::Chat::Session.new(model, vocabulary)
dialogues.each do |turns|
  session.reset
  turns.each do |prompt, target|
    answer = session.reply(prompt)
    puts "#{prompt} → #{answer}"
    abort 'Built-in training dialogue did not converge' unless options[:data] || answer == target
  end
end
TensorRuby::Text::Checkpoint.save(options[:save], model, vocabulary) if options[:save]
