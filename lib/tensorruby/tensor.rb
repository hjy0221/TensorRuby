module TensorRuby
  # Contiguous row-major tensor with one graph node per tensor operation.
  class Tensor
    class Node
      attr_reader :parents

      def initialize(size, parents, &pullback)
        @size, @parents, @pullback = size, parents.freeze, pullback
      end

      def accumulate(pass, index, gradient)
        (pass[self] ||= Array.new(@size, 0.0))[index] += gradient
      end

      def accumulate_vector(pass, gradient)
        existing = pass[self]
        if existing
          gradient.each_with_index { |value, i| existing[i] += value }
        else
          pass[self] = gradient
        end
      end

      def propagate(pass)
        gradient = pass.delete(self)
        @pullback.call(gradient, pass) if gradient && @pullback
      end
    end
    private_constant :Node

    attr_reader :shape, :strides

    def initialize(data, shape: nil)
      inferred, flat = self.class.flatten_data(data)
      set_layout(shape || inferred, flat.length)
      @values = flat.map { |value| Value.wrap(value) }
      @node = Node.new(size, @values.dup) do |gradient, _pass|
        @values.each_with_index { |value, index| value.grad += gradient[index] }
      end
    end

    def self.flatten_data(data)
      return [[], [data]] unless data.is_a?(Array)
      raise ArgumentError, 'empty dimensions unsupported' if data.empty?

      parts = data.map { |value| flatten_data(value) }
      raise ArgumentError, 'ragged tensor' unless parts.all? { |part| part[0] == parts[0][0] }

      [[data.length] + parts[0][0], parts.flat_map(&:last)]
    end

    def self.wrap(data)
      data.is_a?(Tensor) ? data : new(data)
    end

    def values
      Array.new(size) { |index| scalar_at(index) }
    end

    def size; @values.length; end
    def rank; shape.length; end

    def at(*indices)
      raise ArgumentError, 'incorrect index count' unless indices.length == rank
      unless indices.each_with_index.all? { |index, axis| index.is_a?(Integer) && index >= 0 && index < shape[axis] }
        raise IndexError, 'index out of bounds'
      end

      scalar_at(indices.each_with_index.sum { |index, axis| index * strides[axis] })
    end
    alias [] at

    def to_a
      data = flat_data
      build = lambda do |axis, offset|
        return data[offset] if axis == rank

        Array.new(shape[axis]) { |i| build.call(axis + 1, offset + i * strides[axis]) }
      end
      build.call(0, 0)
    end

    def self.broadcast_shape(a, b)
      n = [a.length, b.length].max
      ([1] * (n - a.length) + a).zip([1] * (n - b.length) + b).map do |x, y|
        raise ArgumentError, 'incompatible broadcast shapes' unless x == y || x == 1 || y == 1

        [x, y].max
      end
    end

    def +(other); elementwise(other, :+); end
    def -(other); elementwise(other, :-); end
    def *(other); elementwise(other, :*); end
    def /(other); elementwise(other, :/); end
    def coerce(other); [Tensor.wrap(other), self]; end

    def relu
      data = flat_data
      active = data.map { |value| value > 0 }
      output = data.each_with_index.map { |value, i| active[i] ? value : 0.0 }
      operation(shape, output, [@node]) do |gradient, pass|
        gradient.each_with_index do |value, i|
          @node.accumulate(pass, i, value) if active[i]
        end
      end
    end

    def sum
      reduce_with_gradient(flat_data.sum, Array.new(size, 1.0))
    end

    def mean
      reduce_with_gradient(flat_data.sum / size, Array.new(size, 1.0 / size))
    end

    # Attach a scalar reduction with a row-major derivative per input element.
    def reduce_with_gradient(value, gradient)
      raise ArgumentError, 'gradient length must match tensor size' unless gradient.size == size

      return Value.new(value) unless TensorRuby.grad_enabled?

      scalar_bridge(value, gradient)
    end

    def backward(seed = nil)
      if seed.nil?
        raise ArgumentError, 'non-scalar tensors require a gradient seed' unless size == 1

        weights = [1.0]
      else
        seed = Tensor.wrap(seed)
        raise ArgumentError, 'gradient seed shape must match tensor shape' unless seed.shape == shape

        weights = seed.flat_data
      end
      scalar_bridge(0.0, weights).backward
      self
    end

    def gather_rows(indices)
      unless rank == 2 && indices.is_a?(Array) && !indices.empty? &&
             indices.all? { |i| i.is_a?(Integer) && i >= 0 && i < shape[0] }
        raise ArgumentError, 'expected valid row indices for a rank-2 tensor'
      end

      columns = shape[1]
      data = flat_data
      offsets = indices.map { |i| i * columns }
      output = offsets.flat_map { |offset| data.slice(offset, columns) }
      operation([offsets.size, columns], output, [@node]) do |gradient, pass|
        gradient.each_with_index do |value, index|
          row, column = index.divmod(columns)
          @node.accumulate(pass, offsets[row] + column, value)
        end
      end
    end

    def self.concat_columns(tensors)
      raise ArgumentError, 'at least one tensor is required' if tensors.empty?

      tensors.drop(1).reduce(tensors.first) { |left, right| left.concat_columns(right) }
    end

    def concat_columns(other)
      unless rank == 2 && other.rank == 2 && shape[0] == other.shape[0]
        raise ArgumentError, 'expected rank-2 tensors with equal row counts'
      end

      rows, left_width = shape
      right_width = other.shape[1]
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      output = Array.new(rows) do |row|
        left.slice(row * left_width, left_width) + right.slice(row * right_width, right_width)
      end.flatten
      width = left_width + right_width
      operation([rows, width], output, [@node, right_node]) do |gradient, pass|
        gradient.each_with_index do |value, index|
          row, column = index.divmod(width)
          if column < left_width
            @node.accumulate(pass, row * left_width + column, value)
          else
            right_node.accumulate(pass, row * right_width + column - left_width, value)
          end
        end
      end
    end

    def gelu
      coefficient = Math.sqrt(2.0 / Math::PI)
      data = flat_data
      tanh = data.map { |x| Math.tanh(coefficient * (x + 0.044715 * x**3)) }
      output = data.each_with_index.map { |x, i| 0.5 * x * (1 + tanh[i]) }
      operation(shape, output, [@node]) do |gradient, pass|
        data.each_with_index do |x, i|
          slope = coefficient * (1 + 3 * 0.044715 * x**2)
          derivative = 0.5 * (1 + tanh[i] + x * (1 - tanh[i]**2) * slope)
          @node.accumulate(pass, i, derivative * gradient[i])
        end
      end
    end

    def transpose
      raise ArgumentError, 'transpose requires a rank-2 tensor' unless rank == 2

      rows, columns = shape
      data = flat_data
      indices = Array.new(size) do |index|
        column, row = index.divmod(rows)
        row * columns + column
      end
      operation([columns, rows], indices.map { |index| data[index] }, [@node]) do |gradient, pass|
        indices.each_with_index { |source, i| @node.accumulate(pass, source, gradient[i]) }
      end
    end

    # Normalize each row. Causal mode masks columns beyond the row index.
    def softmax(causal: false)
      raise ArgumentError, 'softmax requires a rank-2 tensor' unless rank == 2

      columns = shape[1]
      probabilities = flat_data.each_slice(columns).with_index.flat_map do |row, index|
        visible = causal ? [index + 1, columns].min : columns
        scores = row.first(visible)
        maximum = scores.max
        weights = scores.map { |score| Math.exp(score - maximum) }
        total = weights.sum
        weights.map { |weight| weight / total } + Array.new(columns - visible, 0.0)
      end
      operation(shape, probabilities, [@node]) do |gradient, pass|
        gradient.each_slice(columns).with_index do |row, row_index|
          offset = row_index * columns
          dot = row.each_with_index.sum { |value, col| value * probabilities[offset + col] }
          row.each_with_index do |value, col|
            index = offset + col
            @node.accumulate(pass, index, probabilities[index] * (value - dot))
          end
        end
      end
    end

    def normalize_rows(eps: 1e-5)
      raise ArgumentError, 'normalization requires a rank-2 tensor' unless rank == 2
      unless eps.is_a?(Numeric) && eps.real? && eps.finite? && eps.positive?
        raise ArgumentError, 'eps must be positive and finite'
      end

      columns = shape[1]
      inverse_stddev = []
      normalized = flat_data.each_slice(columns).flat_map do |row|
        mean = row.sum / columns
        centered = row.map { |value| value - mean }
        variance = centered.sum { |value| value * value } / columns
        inverse = 1.0 / Math.sqrt(variance + eps)
        inverse_stddev << inverse
        centered.map { |value| value * inverse }
      end
      operation(shape, normalized, [@node]) do |gradient, pass|
        gradient.each_slice(columns).with_index do |row, row_index|
          offset = row_index * columns
          mean = row.sum / columns
          dot_mean = row.each_with_index.sum { |value, col| value * normalized[offset + col] } / columns
          row.each_with_index do |value, col|
            index = offset + col
            derivative = inverse_stddev[row_index] * (value - mean - normalized[index] * dot_mean)
            @node.accumulate(pass, index, derivative)
          end
        end
      end
    end

    def matmul(other)
      other = Tensor.wrap(other)
      raise ArgumentError, 'matmul requires rank-2 tensors' unless rank == 2 && other.rank == 2
      raise ArgumentError, 'matmul dimension mismatch' unless shape[1] == other.shape[0]

      rows, inner = shape
      cols = other.shape[1]
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      output = CPU.matmul(left, right, rows, inner, cols)
      operation([rows, cols], output, [@node, right_node]) do |gradient, pass|
        grad_left, grad_right = CPU.matmul_backward(left, right, gradient, rows, inner, cols)
        @node.accumulate_vector(pass, grad_left)
        right_node.accumulate_vector(pass, grad_right)
      end
    end

    protected

    def graph_node; @node; end

    def flat_data
      @values.each_with_index.map { |value, i| value ? value.data : @data[i] }
    end

    def broadcast_offset(coordinates)
      coordinates.last(rank).each_with_index.sum do |coordinate, axis|
        shape[axis] == 1 ? 0 : coordinate * strides[axis]
      end
    end

    private

    def scalar_bridge(value, gradient)
      weights = gradient.dup.freeze
      Value.new(value, parents: [@node], tensor_bridge: true) do |upstream, pass|
        weights.each_with_index { |weight, i| @node.accumulate(pass, i, upstream * weight) }
      end
    end

    def elementwise(other, operator)
      other = Tensor.wrap(other)
      result_shape = Tensor.broadcast_shape(shape, other.shape)
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      offsets = Array.new(result_shape.inject(1, :*)) do |index|
        coordinates = result_shape.reverse.map do |dimension|
          index, coordinate = index.divmod(dimension)
          coordinate
        end.reverse
        [broadcast_offset(coordinates), other.broadcast_offset(coordinates)]
      end
      output = offsets.map { |a, b| left[a].public_send(operator, right[b]) }
      operation(result_shape, output, [@node, right_node]) do |gradient, pass|
        offsets.each_with_index do |(a, b), i|
          x, y = left[a], right[b]
          da, db = case operator
                   when :+ then [1.0, 1.0]
                   when :- then [1.0, -1.0]
                   when :* then [y, x]
                   when :/ then [1.0 / y, -x / (y * y)]
                   end
          @node.accumulate(pass, a, gradient[i] * da)
          right_node.accumulate(pass, b, gradient[i] * db)
        end
      end
    end

    def set_layout(shape, size)
      @shape = shape.dup.freeze
      unless @shape.all? { |dimension| dimension.is_a?(Integer) && dimension.positive? }
        raise ArgumentError, 'dimensions must be positive integers'
      end
      raise ArgumentError, 'shape does not match data' unless @shape.inject(1, :*) == size

      stride = 1
      @strides = @shape.reverse.map do |dimension|
        current = stride
        stride *= dimension
        current
      end.reverse.freeze
    end

    def operation(shape, data, parents, &pullback)
      result = Tensor.allocate
      if TensorRuby.grad_enabled?
        result.initialize_operation(shape, data, parents, &pullback)
      else
        result.initialize_operation(shape, data, [])
      end
      result
    end

    protected def initialize_operation(shape, data, parents, &pullback)
      set_layout(shape, data.length)
      @data = data.freeze
      @values = Array.new(data.length)
      @node = Node.new(size, parents) do |gradient, pass|
        @last_gradient = gradient
        @values.each_with_index { |value, i| value.grad = gradient[i] if value }
        pullback.call(gradient, pass) if pullback
      end
    end

    def scalar_at(index)
      @values[index] ||= begin
        value = Value.new(@data[index], parents: [@node], tensor_bridge: true) do |gradient, pass|
          @node.accumulate(pass, index, gradient)
        end
        value.grad = @last_gradient[index] if @last_gradient
        value
      end
    end
  end
end
