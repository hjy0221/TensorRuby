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

    def matmul(other)
      other = Tensor.wrap(other)
      raise ArgumentError, 'matmul requires rank-2 tensors' unless rank == 2 && other.rank == 2
      raise ArgumentError, 'matmul dimension mismatch' unless shape[1] == other.shape[0]

      rows, inner = shape
      cols = other.shape[1]
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      output = Array.new(rows * cols) do |index|
        row, col = index.divmod(cols)
        inner.times.sum { |k| left[row * inner + k] * right[k * cols + col] }
      end
      operation([rows, cols], output, [@node, right_node]) do |gradient, pass|
        gradient.each_with_index do |upstream, index|
          row, col = index.divmod(cols)
          inner.times do |k|
            a = row * inner + k
            b = k * cols + col
            @node.accumulate(pass, a, upstream * right[b])
            right_node.accumulate(pass, b, upstream * left[a])
          end
        end
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
