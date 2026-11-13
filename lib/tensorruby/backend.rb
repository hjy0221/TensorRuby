module TensorRuby
  module Backend
    # Reference alternative: pack columns into contiguous rows before dot products.
    module PackedCPU
      def self.matmul(left, right, rows, inner, columns)
        packed = transpose(right, inner, columns)
        Array.new(rows * columns) do |index|
          row, column = index.divmod(columns)
          a = row * inner
          b = column * inner
          inner.times.sum { |k| left[a + k] * packed[b + k] }
        end
      end

      def self.matmul_backward(left, right, gradient, rows, inner, columns)
        [
          matmul(gradient, transpose(right, inner, columns), rows, columns, inner),
          matmul(transpose(left, rows, inner), gradient, inner, rows, columns)
        ]
      end

      def self.transpose(data, rows, columns)
        Array.new(data.size) do |index|
          column, row = index.divmod(rows)
          data[row * columns + column]
        end
      end
      private_class_method :transpose
    end

    # Returns per-kernel measurements for the operations executed inside the block.
    def self.profile
      previous = Thread.current[:tensorruby_backend_profile]
      measurements = {}
      Thread.current[:tensorruby_backend_profile] = measurements
      yield
      measurements.values
    ensure
      Thread.current[:tensorruby_backend_profile] = previous
    end

    def self.measure(backend, phase, rows, inner, columns)
      measurements = Thread.current[:tensorruby_backend_profile]
      return yield unless measurements

      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      result = yield
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
      key = [backend, phase, rows, inner, columns]
      entry = measurements[key] ||= {
        backend: backend, phase: phase, shape: [rows, inner, columns],
        calls: 0, seconds: 0.0, estimated_flops: 0
      }
      entry[:calls] += 1
      entry[:seconds] += elapsed
      entry[:estimated_flops] += (phase == :forward ? 2 : 4) * rows * inner * columns
      result
    end

    def self.current
      Thread.current[:tensorruby_matmul_backend] || CPU
    end

    def self.with(backend)
      previous = Thread.current[:tensorruby_matmul_backend]
      unless backend.respond_to?(:matmul) && backend.respond_to?(:matmul_backward)
        raise ArgumentError, 'backend must implement matmul and matmul_backward'
      end

      Thread.current[:tensorruby_matmul_backend] = backend
      yield
    ensure
      Thread.current[:tensorruby_matmul_backend] = previous
    end
  end
end
