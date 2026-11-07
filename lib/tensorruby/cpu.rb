module TensorRuby
  # Numerical kernels for flat, contiguous row-major Float arrays.
  # Tensor owns shape validation and the autodiff graph.
  module CPU
    def self.matmul(left, right, rows, inner, columns)
      Array.new(rows * columns) do |index|
        row, column = index.divmod(columns)
        left_offset = row * inner
        inner.times.sum { |k| left[left_offset + k] * right[k * columns + column] }
      end
    end

    def self.matmul_backward(left, right, gradient, rows, inner, columns)
      grad_left = Array.new(left.size, 0.0)
      grad_right = Array.new(right.size, 0.0)

      rows.times do |row|
        left_offset = row * inner
        output_offset = row * columns
        inner.times do |k|
          left_index = left_offset + k
          right_offset = k * columns
          left_value = left[left_index]
          contribution = 0.0
          columns.times do |column|
            upstream = gradient[output_offset + column]
            right_index = right_offset + column
            contribution += upstream * right[right_index]
            grad_right[right_index] += upstream * left_value
          end
          grad_left[left_index] = contribution
        end
      end
      [grad_left, grad_right]
    end
  end
end
