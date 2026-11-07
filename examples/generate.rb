$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'optparse'

options = { tokens: 24 }
OptionParser.new do |parser|
  parser.banner = 'Usage: ruby examples/generate.rb [--chat] [--tokens N] CHECKPOINT.json "prompt"'
  parser.on('--chat') { options[:chat] = true }
  parser.on('--tokens N', Integer) { |count| options[:tokens] = count }
end.parse!
abort 'provide a checkpoint, a nonempty prompt, and a positive token limit' unless
  ARGV.size == 2 && !ARGV.last.strip.empty? && options[:tokens].positive?

path, prompt = ARGV
model, vocabulary = TensorRuby::Text::Checkpoint.load(path)
model.eval
if options[:chat]
  session = TensorRuby::Chat::Session.new(model, vocabulary, max_reply_tokens: options[:tokens])
  puts session.reply(prompt)
else
  input = vocabulary.encode(prompt)
  output = model.generate(input, max_new_tokens: options[:tokens])
  puts vocabulary.decode(output)
end
