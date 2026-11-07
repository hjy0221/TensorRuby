$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'optparse'

options = { tokens: 24 }
OptionParser.new do |parser|
  parser.banner = 'Usage: generate.rb [--chat | --interactive] [--tokens N] CHECKPOINT.json [PROMPT]'
  parser.on('--chat') { options[:chat] = true }
  parser.on('--interactive') { options[:interactive] = true }
  parser.on('--tokens N', Integer) { |count| options[:tokens] = count }
end.parse!

path, prompt = ARGV
expected_arguments = options[:interactive] ? 1 : 2
unless ARGV.size == expected_arguments && options[:tokens].positive? &&
       (options[:interactive] || !prompt.strip.empty?)
  abort 'provide a checkpoint, a prompt unless interactive, and a positive token limit'
end

model, vocabulary = TensorRuby::Text::Checkpoint.load(path)
model.eval
if options[:interactive]
  session = TensorRuby::Chat::Session.new(model, vocabulary, max_reply_tokens: options[:tokens])
  puts 'Enter a message. /reset clears context and /quit exits.'
  STDIN.each_line do |line|
    message = line.strip
    case message
    when '/quit' then break
    when '/reset' then session.reset
    when '' then next
    else puts session.reply(message)
    end
  end
elsif options[:chat]
  session = TensorRuby::Chat::Session.new(model, vocabulary, max_reply_tokens: options[:tokens])
  puts session.reply(prompt)
else
  input = vocabulary.encode(prompt)
  puts vocabulary.decode(model.generate(input, max_new_tokens: options[:tokens]))
end
