require 'tensorruby'

iterations = Integer(ENV.fetch('ITERATIONS', '10'))
abort 'ITERATIONS must be positive' unless iterations.positive?

puts "TensorRuby #{TensorRuby::VERSION} | #{RUBY_DESCRIPTION} | #{RUBY_PLATFORM}"
puts "OS: #{`uname -sr`.strip} | dtype: Float (binary64) | seed: 42 | iterations: #{iterations}"
puts 'Scope: forward + backward, including tensor graph and gradient reset'
random = Random.new(42)
matrix = ->(rows, columns) { TensorRuby.tensor(Array.new(rows) { Array.new(columns) { random.rand - 0.5 } }) }
left, right = matrix.call(48, 48), matrix.call(48, 48)
input = matrix.call(12, 16)
block = TensorRuby::NN::TransformerBlock.new(16, heads: 4, hidden_features: 32, random: random)

backends = [TensorRuby::CPU, TensorRuby::Backend::PackedCPU]
backends << TensorRuby::Backend::Accelerate if TensorRuby::Backend::Accelerate.available?

workloads = {
  'matmul 48×48' => [left.values + right.values, -> { left.matmul(right) }],
  'Transformer 12×16' => [block.parameters + input.values, -> { block.call(input) }]
}
workloads.each do |label, (parameters, forward)|
  backends.each do |backend|
    TensorRuby::Backend.with(backend) do
      run = -> { parameters.each(&:zero_grad); forward.call.sum.backward }
      2.times { run.call }
      GC.start
      start_objects = GC.stat(:total_allocated_objects)
      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      iterations.times { run.call }
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
      allocations = GC.stat(:total_allocated_objects) - start_objects
      puts format('%s | %s | %.3f ms | %.0f objects', label, backend,
                  elapsed * 1000 / iterations, allocations.fdiv(iterations))
    end
  end
end

backends.each do |backend|
  block.zero_grad
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  report = TensorRuby::Backend.profile do
    TensorRuby::Backend.with(backend) { block.call(input).sum.backward }
  end
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  puts format('%s | Transformer profile: %.3f ms total', backend, elapsed * 1000)
  report.each do |entry|
    puts format('  %s %s | %d calls | %.3f ms | %d estimated FLOPs',
                entry[:phase], entry[:shape].join('×'), entry[:calls],
                entry[:seconds] * 1000, entry[:estimated_flops])
  end
end
