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

criterion = TensorRuby::NN::CrossEntropyLoss.new(ignore_index: TensorRuby::Chat::Vocabulary::IGNORE)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.015)
random = Random.new(7)
measure = lambda do
  TensorRuby.no_grad { examples.sum { |input, targets| criterion.call(model.call(input), targets).data } / examples.size }
end
before = measure.call
model.train
options[:epochs].times do
  examples.shuffle(random: random).each do |input, targets|
    optimizer.zero_grad
    criterion.call(model.call(input), targets).backward
    optimizer.step
  end
end
model.eval
after = measure.call
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
