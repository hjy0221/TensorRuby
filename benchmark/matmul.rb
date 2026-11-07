require 'tensorruby'

size = Integer(ENV.fetch('SIZE', '32'))
iterations = Integer(ENV.fetch('ITERATIONS', '10'))
abort 'SIZE and ITERATIONS must be positive' unless size.positive? && iterations.positive?

random = Random.new(42)
matrix = -> { TensorRuby.tensor(Array.new(size) { Array.new(size) { random.rand - 0.5 } }) }
left, right = matrix.call, matrix.call

def measure(label, iterations)
  2.times { yield }
  GC.start
  allocated = GC.stat(:total_allocated_objects)
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  iterations.times { yield }
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  objects = GC.stat(:total_allocated_objects) - allocated
  puts format('%-20s %9.3f ms/iteration  %9.1f objects/iteration',
              label, 1000 * elapsed / iterations, objects.fdiv(iterations))
end

puts "Ruby #{RUBY_VERSION} | TensorRuby #{TensorRuby::VERSION} | #{size}×#{size} | #{iterations} iterations"
measure('forward (no_grad)', iterations) { TensorRuby.no_grad { left.matmul(right) } }
parameters = left.values + right.values
measure('forward + backward', iterations) do
  parameters.each(&:zero_grad)
  left.matmul(right).sum.backward
end
