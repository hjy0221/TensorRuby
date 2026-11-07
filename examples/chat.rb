$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'json'
require 'optparse'

chat = TensorRuby::Chat
nn = TensorRuby::NN
options = { epochs: 100 }
OptionParser.new do |parser|
  parser.banner = 'Usage: ruby examples/chat.rb [--interactive] [--data dialogues.json] [--epochs N]'
  parser.on('--interactive') { options[:interactive] = true }
  parser.on('--data FILE') { |file| options[:data] = file }
  parser.on('--epochs N', Integer) { |count| options[:epochs] = count }
end.parse!
abort 'epochs must be positive' unless options[:epochs].positive?
dialogues = [
  [['오늘 뭐 할까?', '산책을 해봐요.'], ['준비물은?', '편한 신발을 챙겨요.']],
  [['오늘 뭐 만들까?', '커피를 내려봐요.'], ['준비물은?', '원두와 물을 챙겨요.']],
  [['안녕', '반가워요!'], ['이름이 뭐야?', '저는 텐서루비예요.']],
  [['散歩に行こうかな', '散歩しましょう。'], ['何を用意する？', '歩きやすい靴を履きましょう。']],
  [['コーヒーを淹れようかな', 'コーヒーを淹れましょう。'], ['何を用意する？', '豆と水を用意しましょう。']],
  [['こんにちは', 'はじめまして！'], ['名前は？', 'テンソルルビーです。']],
  [['I want to go for a walk', 'Let us take a walk.'], ['What should I bring?', 'Wear comfortable shoes.']],
  [['I want to make coffee', 'Let us brew coffee.'], ['What should I bring?', 'Bring coffee beans and water.']],
  [['Hello', 'Nice to meet you!'], ['What is your name?', 'I am TensorRuby.']]
]
dialogues = JSON.parse(File.read(options[:data])) if options[:data]
valid = dialogues.is_a?(Array) && !dialogues.empty? && dialogues.all? do |dialogue|
  dialogue.is_a?(Array) && !dialogue.empty? && dialogue.all? do |turn|
    turn.is_a?(Array) && turn.size == 2 && turn.all? { |text| text.is_a?(String) && !text.strip.empty? }
  end
end
abort 'expected a JSON array of dialogues containing [user, assistant] string pairs' unless valid
vocabulary = chat::Vocabulary.new(dialogues.flatten)
examples = dialogues.map { |turns| vocabulary.training_example(turns) }
context = examples.map { |input, _| input.size }.max + 2
random = Random.new(21)
model = nn::GPT.new(vocab_size: vocabulary.size, context_size: context, features: 16,
                    heads: 4, layers: 1, random: random)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.015)
criterion = nn::CrossEntropyLoss.new(ignore_index: chat::Vocabulary::IGNORE)

options[:epochs].times do |epoch|
  examples.shuffle(random: random).each do |input, target|
    optimizer.zero_grad
    criterion.call(model.call(input), target).backward
    optimizer.step
  end
  puts "Training epoch #{epoch + 1}/#{options[:epochs]}" if ((epoch + 1) % 20).zero?
end

session = chat::Session.new(model.eval, vocabulary)
dialogues.each do |turns|
  session.reset
  turns.each do |question, expected|
    answer = session.reply(question)
    puts "사용자: #{question}\n모델: #{answer}"
    abort "Expected #{expected.inspect}" unless options[:data] || answer == expected
  end
end

if options[:interactive]
  session.reset
  puts '학습한 질문을 입력하세요. /reset은 대화 초기화, /quit은 종료입니다.'
  while (line = STDIN.gets)
    message = line.strip
    break if message == '/quit'
    if message == '/reset'
      session.reset
      next
    end

    begin
      puts "모델: #{session.reply(message)}"
    rescue ArgumentError => error
      warn error.message
    end
  end
end
