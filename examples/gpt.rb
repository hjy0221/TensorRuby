$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

nn = TensorRuby::NN
model = nn::GPT.new(vocab_size: 4, context_size: 4, features: 8, heads: 2, layers: 1, random: Random.new(19))
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.02)
trainer = TensorRuby::Training::Trainer.new(model, optimizer: optimizer, loss: nn::CrossEntropyLoss.new)
sequence = [0, 1, 2, 3, 0, 1, 2, 3]
windows = 4.times.map { |offset| sequence.slice(offset, 5).each_cons(2).to_a.transpose }

50.times { |epoch| trainer.train_epoch([windows[epoch % windows.size]]) }

generated = model.eval.generate([0], max_new_tokens: 7)
puts "Generated token IDs: #{generated.inspect}"
abort 'Next-token training failed' unless generated == sequence
