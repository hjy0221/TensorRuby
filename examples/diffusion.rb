$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'

nn = TensorRuby::NN
random = Random.new(23)
schedule = TensorRuby::Diffusion::Schedule.new(steps: 24, beta_start: 0.02, beta_end: 0.25)
model = nn::Sequential.new(
  nn::Linear.new(3, 24, random: random),
  nn::ReLU.new,
  nn::Linear.new(24, 24, random: random),
  nn::ReLU.new,
  nn::Linear.new(24, 2, random: random)
)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.005)
loss_fn = nn::MSELoss.new
predict_noise = lambda do |noisy, step|
  time = step.fdiv(schedule.steps - 1)
  model.call(TensorRuby.tensor(noisy.to_a.map { |point| point + [time] }))
end

make_batch = lambda do
  clean = TensorRuby.tensor(Array.new(24) do
    direction = random.rand(2).zero? ? -1 : 1
    [direction * 2.0 + random.rand * 0.2 - 0.1, direction + random.rand * 0.2 - 0.1]
  end)
  noise = TensorRuby::Diffusion::Schedule.normal(clean.shape, random: random)
  step = random.rand(schedule.steps)
  [schedule.add_noise(clean, noise, step), noise, step]
end

# A training example carries the noisy input and its timestep together.
example = lambda do
  noisy, noise, step = make_batch.call
  [[noisy, step], noise]
end
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: loss_fn, forward: ->(input) { predict_noise.call(*input) }
)
validation = Array.new(12) { example.call }
initial_loss = trainer.evaluate(validation)
500.times { trainer.train_epoch([example.call]) }
final_loss = trainer.evaluate(validation)
model.eval
points = schedule.sample([8, 2], random: random, &predict_noise).to_a
puts format('Held-out noise MSE: %.4f -> %.4f', initial_loss, final_loss)
puts "Generated points: #{points.map { |point| point.map { |value| value.round(2) } }.inspect}"
abort 'Diffusion training failed' unless final_loss < initial_loss && points.flatten.all?(&:finite?)
