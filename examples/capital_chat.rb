$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'json'
require 'optparse'

options = { epochs: 32 }
OptionParser.new do |parser|
  parser.on('--data FILE') { |value| options[:data] = value }
  parser.on('--epochs N', Integer) { |value| options[:epochs] = value }
  parser.on('--save FILE') { |value| options[:save] = value }
  parser.on('--load FILE') { |value| options[:load] = value }
  parser.on('--interactive') { options[:interactive] = true }
end.parse!
abort 'epochs must be positive' unless options[:epochs].positive?

facts = JSON.parse(File.read(options[:data] || File.join(__dir__, 'data', 'capitals.json')))
phrases = {
  'ko' => ['수도 알려줘', '수도 말해줘', '알려줘 수도'],
  'ja' => ['首都 教えて', '首都 どこ', '教えて 首都'],
  'en' => ['capital please', 'capital city', 'please capital']
}
training = {}
evaluation = {}
phrases.each do |language, (first, second, unseen)|
  training[language] = facts.fetch(language).flat_map do |country, capital|
    [first, second].map { |suffix| ["#{country} #{suffix}", capital] }
  end
  evaluation[language] = facts.fetch(language).map { |country, capital| ["#{country} #{unseen}", capital] }
end

chat = TensorRuby::Chat
nn = TensorRuby::NN
vocabulary = chat::Vocabulary.new(training.values.flatten)
examples = training.values.flatten(1).map { |pair| vocabulary.training_example([pair]) }
context = examples.map { |input, _| input.size }.max + 1
random = Random.new(51)
model = nn::GPT.new(vocab_size: vocabulary.size, context_size: context, features: 16,
                    heads: 4, layers: 1, random: random)
parameters = model.parameters
if options[:load]
  checkpoint = JSON.parse(File.read(options[:load]))
  unless checkpoint['data'] == facts && checkpoint['weights'].is_a?(Array) &&
         checkpoint['weights'].size == parameters.size &&
         checkpoint['weights'].all? { |weight| weight.is_a?(Numeric) && weight.finite? }
    abort 'checkpoint does not match this model and dataset'
  end
  parameters.zip(checkpoint['weights']).each { |parameter, weight| parameter.data = weight.to_f }
else
  optimizer = TensorRuby::Optim::Adam.new(parameters, lr: 0.008)
  criterion = nn::CrossEntropyLoss.new(ignore_index: chat::Vocabulary::IGNORE)
  trainer = TensorRuby::Training::Trainer.new(model, optimizer: optimizer, loss: criterion)
  options[:epochs].times do |epoch|
    mean = trainer.train_epoch(examples.shuffle(random: random))
    puts format('Epoch %d loss %.4f', epoch + 1, mean) if ((epoch + 1) % 8).zero?
  end
end
if options[:save]
  File.write(options[:save], JSON.generate('data' => facts, 'weights' => parameters.map(&:data)))
end

session = chat::Session.new(model.eval, vocabulary)
phrases.each_key do |language|
  [training, evaluation].zip(%w[trained reordered]).each do |split, label|
    pairs = split.fetch(language)
    correct = pairs.count { |question, capital| session.reset.reply(question) == capital }
    puts "#{language} #{label}: #{correct}/#{pairs.size}"
  end
  question, = evaluation.fetch(language).first
  puts "#{question} → #{session.reset.reply(question)}"
end

if options[:interactive]
  puts 'Enter a capital question, or /quit. Examples: 프랑스 수도 알려줘 | フランス 首都 教えて | France capital please'
  STDIN.each_line do |line|
    question = line.strip
    break if question == '/quit'

    begin
      puts session.reset.reply(question)
    rescue ArgumentError => error
      warn error.message
    end
  end
end
