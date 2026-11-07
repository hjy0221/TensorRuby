$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'tensorruby'
require 'zlib'

# A small conditional diffusion model generates three sprite attributes.
# The pixel renderer supplies the silhouette; it does not predict pixels.
species = %w[dino wolf bird]
requested = ARGV.fetch(0, 'all')
abort "Choose all, #{species.join(', ')}" unless requested == 'all' || species.include?(requested)
path = ARGV.fetch(1, 'monster-sprites.png')
random = Random.new(Integer(ARGV.fetch(2, '37')))
schedule = TensorRuby::Diffusion::Schedule.new(steps: 24, beta_start: 0.02, beta_end: 0.25)
nn = TensorRuby::NN
model = nn::Sequential.new(
  nn::Linear.new(7, 20, random: random),
  nn::GELU.new,
  nn::Linear.new(20, 3, random: random)
)
optimizer = TensorRuby::Optim::Adam.new(model.parameters, lr: 0.008)
loss = nn::MSELoss.new
centers = [[0.6, 0.3, -0.4], [-0.5, 0.2, 0.4], [0.2, -0.5, 0.3]]
predict = lambda do |noisy, step, labels|
  conditions = labels.map do |label|
    [step.fdiv(schedule.steps - 1)] + species.each_index.map { |i| i == label ? 1.0 : 0.0 }
  end
  model.call(noisy.concat_columns(TensorRuby.tensor(conditions)))
end
batch = lambda do
  labels = Array.new(12) { random.rand(species.size) }
  clean = TensorRuby.tensor(labels.map { |label| centers[label].map { |value| value + 0.6 * (random.rand - 0.5) } })
  noise = TensorRuby::Diffusion::Schedule.normal(clean.shape, random: random)
  step = random.rand(schedule.steps)
  [schedule.add_noise(clean, noise, step), noise, step, labels]
end

# A training example carries the noisy input and its timestep together.
example = lambda do
  noisy, noise, step, labels = batch.call
  [[noisy, step, labels], noise]
end
trainer = TensorRuby::Training::Trainer.new(
  model, optimizer: optimizer, loss: loss, forward: ->(input) { predict.call(*input) }
)
validation = Array.new(10) { example.call }
initial_loss = trainer.evaluate(validation)
300.times { trainer.train_epoch([example.call]) }
final_loss = trainer.evaluate(validation)
model.eval
abort 'Noise predictor did not improve' unless final_loss.finite? && final_loss < initial_loss

labels = requested == 'all' ? [0, 1, 2, 0, 1, 2] : Array.new(4, species.index(requested))
attributes = schedule.sample([labels.size, 3], random: random) { |noisy, step| predict.call(noisy, step, labels) }.to_a
abort 'Sampler returned nonfinite attributes' unless attributes.flatten.all?(&:finite?)

sprites = [
  %w[
    .......####.....
    ......#HHHH##...
    .....#HHHHHHH#..
    .....#HHEEHHHH#.
    .....#HHEPHHHH#.
    .....#HHHHHHHHH#
    .....#HHHHHHHHH#
    ......#HH#MM###.
    .....#HHHH##T#..
    .#..#HHHHBHH#...
    #H##HHHHBB#H#...
    #HHHHHHHBB###...
    .##HHHHHBB#.....
    ...#HHHHHH#.....
    ...#H##HHH##....
    ...###.#####....
  ],
  %w[
    ..##.......##...
    ..#B#.....#B#...
    ..#HB#...#BH#...
    ...#HH###HH#....
    ...#HHHBHHH#....
    ..#HEEHHEEHH#...
    ..#HEPHHEPHH#...
    .#HHHHBBHHHHH#..
    ..##BBBBBBB##...
    ....#BBPBB#.....
    .....#BMB#......
    ....#HHBHH#..##.
    ...#HHHBHHH##H#.
    ...#HHHBHHHHHH#.
    ....#H###H###...
    ....###.###.....
  ],
  %w[
    .......##.......
    ......#HH##.....
    .....#HHHHH#....
    ....#HHHHHHH#...
    ...#HHHHEEHHH#..
    ...#HHHHEPHHH#..
    ...#HHHHHHBBB##.
    ..#HHHHHHBBBBB#.
    .#HHHBBBBH####..
    #HHHHBBBBH#.....
    #HHHHHHHHHH#....
    .##HHHHHHHH#....
    ...#HHHHHH#.....
    ....##HH##......
    .....#BB#.......
    ....######......
  ]
]
colors = [
  [[239, 139, 55], [255, 205, 127]],
  [[102, 148, 203], [213, 232, 243]],
  [[238, 118, 162], [254, 216, 106]]
]
shade = lambda do |color, factor|
  color.map { |channel| (channel * factor).round.clamp(0, 255) }
end
tiles = labels.zip(attributes).map do |label, (hue, contrast, expression)|
  base, accent = colors[label]
  base = [base[0] + 22 * hue, base[1], base[2] - 18 * hue].map { |channel| channel.clamp(0, 255) }
  palette = {
    '.' => [244, 240, 225], '#' => [35, 27, 36], 'H' => shade.call(base, 1 + 0.16 * contrast),
    'B' => shade.call(accent, 1 - 0.12 * contrast), 'E' => [255, 250, 235],
    'P' => expression.positive? ? [33, 109, 89] : [46, 45, 69],
    'M' => [148, 49, 57], 'T' => [255, 246, 215]
  }
  sprites[label].map { |row| row.chars.map { |pixel| palette.fetch(pixel) } }
end

columns = requested == 'all' ? 3 : 2
scale = 6
tile_size = 20
width, height = columns * tile_size * scale, 2 * tile_size * scale
scanlines = Array.new(height) do |y|
  row = Array.new(width) do |x|
    tile = tiles[(y / (tile_size * scale)) * columns + x / (tile_size * scale)]
    row_index = (y / scale) % tile_size - 2
    column_index = (x / scale) % tile_size - 2
    if row_index.between?(0, 15) && column_index.between?(0, 15)
      tile.fetch(row_index).fetch(column_index)
    else
      [244, 240, 225]
    end
  end
  [0, *row.flatten].pack('C*')
end.join
chunk = lambda do |type, data|
  [data.bytesize].pack('N') + type + data + [Zlib.crc32(type + data)].pack('N')
end
png = "\x89PNG\r\n\x1a\n".b
png << chunk.call('IHDR', [width, height, 8, 2, 0, 0, 0].pack('NNC5'))
png << chunk.call('IDAT', Zlib.deflate(scanlines))
png << chunk.call('IEND', ''.b)
File.binwrite(path, png)
puts format('Held-out attribute noise MSE: %.4f -> %.4f', initial_loss, final_loss)
puts "Generated #{tiles.size} monster sprites: #{path}"
