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

validation = Array.new(12) { make_batch.call }
validation_loss = lambda do
  TensorRuby.no_grad do
    validation.sum { |noisy, noise, step| loss_fn.call(predict_noise.call(noisy, step), noise).data } / validation.size
  end
end
initial = validation_loss.call
500.times do
  noisy, noise, step = make_batch.call
  optimizer.zero_grad
  loss_fn.call(predict_noise.call(noisy, step), noise).backward
  optimizer.step
end

model.eval
final = validation_loss.call
points = schedule.sample([8, 2], random: random, &predict_noise).to_a
puts format('Held-out noise MSE: %.4f -> %.4f', initial, final)
puts "Generated points: #{points.map { |point| point.map { |value| value.round(2) } }.inspect}"
abort 'Diffusion training failed' unless final < initial && points.flatten.all?(&:finite?)
