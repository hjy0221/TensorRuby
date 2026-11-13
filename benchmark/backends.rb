require 'tensorruby'

iterations = 10
random = Random.new(42)
matrix = ->(rows, columns) { TensorRuby.tensor(Array.new(rows) { Array.new(columns) { random.rand - 0.5 } }) }
left, right = matrix.call(48, 48), matrix.call(48, 48)
input = matrix.call(12, 16)
block = TensorRuby::NN::TransformerBlock.new(16, heads: 4, hidden_features: 32, random: random)

workloads = {
  'matmul 48×48' => [left.values + right.values, -> { left.matmul(right) }],
  'Transformer 12×16' => [block.parameters + input.values, -> { block.call(input) }]
}
workloads.each do |label, (parameters, forward)|
  [TensorRuby::CPU, TensorRuby::Backend::PackedCPU].each do |backend|
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
