$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'zlib'

side = 6
pixels = side * side
random = Random.new(31)
schedule = TensorRuby::Diffusion::Schedule.new(steps: 24, beta_start: 0.02, beta_end: 0.25)
nn = TensorRuby::NN
model = nn::Sequential.new(
  nn::Linear.new(pixels + 1, 24, random: random),
  nn::GELU.new,
  nn::Linear.new(24, pixels, random: random)
)
loss_fn = nn::MSELoss.new
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.008)

predict_noise = lambda do |noisy, step|
  time = TensorRuby.tensor(Array.new(noisy.shape[0]) { [step.fdiv(schedule.steps - 1)] })
  model.call(noisy.concat_columns(time))
end
make_batch = lambda do
  patterns = Array.new(8) do
    horizontal = random.rand(2).zero?
    stripe = random.rand(1..3)
    Array.new(pixels) do |i|
      row, column = i.divmod(side)
      coordinate = horizontal ? row : column
      (stripe...(stripe + 2)).cover?(coordinate) ? 1.0 : -1.0
    end
  end
  clean = TensorRuby.tensor(patterns)
  noise = TensorRuby::Diffusion::Schedule.normal(clean.shape, random: random)
  step = random.rand(schedule.steps)
  [schedule.add_noise(clean, noise, step), noise, step]
end

validation = Array.new(12) { make_batch.call }
evaluate = lambda do
  TensorRuby.no_grad do
    validation.sum { |noisy, noise, step| loss_fn.call(predict_noise.call(noisy, step), noise).data } / validation.size
  end
end

initial_loss = evaluate.call
400.times do
  noisy, noise, step = make_batch.call
  optimizer.zero_grad
  loss_fn.call(predict_noise.call(noisy, step), noise).backward
  optimizer.step
end

model.eval
final_loss = evaluate.call
images = schedule.sample([8, pixels], random: random, &predict_noise).to_a
unless final_loss < initial_loss && images.flatten.all?(&:finite?)
  abort 'Image diffusion training failed'
end

# Write a grayscale PNG contact sheet, enlarging each pixel for inspection.
scale = 12
tile = side * scale
width, height = 4 * tile, 2 * tile
scanlines = Array.new(height) do |y|
  row = Array.new(width) do |x|
    image = images[(y / tile) * 4 + x / tile]
    pixel = image[((y % tile) / scale) * side + (x % tile) / scale]
    ((pixel.clamp(-1.0, 1.0) + 1.0) * 127.5).round
  end
  [0, *row].pack('C*')
end.join
chunk = lambda do |type, data|
  [data.bytesize].pack('N') + type + data + [Zlib.crc32(type + data)].pack('N')
end
png = "\x89PNG\r\n\x1a\n".b
png << chunk.call('IHDR', [width, height, 8, 0, 0, 0, 0].pack('NNC5'))
png << chunk.call('IDAT', Zlib.deflate(scanlines))
png << chunk.call('IEND', ''.b)
path = ARGV.fetch(0, 'diffusion-samples.png')
File.binwrite(path, png)
puts format('Held-out image noise MSE: %.4f -> %.4f', initial_loss, final_loss)
puts "Generated #{images.size} images: #{path}"
