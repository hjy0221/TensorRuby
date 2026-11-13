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
