require 'tensorruby'

input = TensorRuby.tensor(Array.new(4096) { |i| i.fdiv(4096) - 0.5 })
paths = {
  separate: -> { (input * 2 + 0.25).relu },
  fused: -> { input.affine_relu(scale: 2, bias: 0.25) }
}

paths.each do |name, forward|
  TensorRuby.no_grad do
    3.times { forward.call }
    GC.start
    allocations = GC.stat(:total_allocated_objects)
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    30.times { forward.call }
    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
    objects = GC.stat(:total_allocated_objects) - allocations
    puts format('%s: %.3f ms, %.1f objects per iteration', name, elapsed * 1000 / 30, objects.fdiv(30))
  end
end
