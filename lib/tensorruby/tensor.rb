module TensorRuby
  # Strided tensor with one graph node per tensor operation.
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

    class Storage
      def initialize(data, values)
        @data, @values = data, values
      end

      def read(index)
        value = @values[index]
        value ? value.data : @data[index]
      end

      def write(index, data)
        value = @values[index]
        if value
          value.data = data
        else
          @data[index] = data
        end
      end
    end

    class ViewValue < Value
      def initialize(storage, offset, **options, &backward)
        @storage, @offset = storage, offset
        super(storage.read(offset), **options, &backward)
      end

      def data
        @storage.read(@offset)
      end

      def data=(value)
        @storage.write(@offset, value)
      end
    end
    private_constant :Storage, :ViewValue

    attr_reader :shape, :strides

    def initialize(data, shape: nil)
      inferred, flat = self.class.flatten_data(data)
      set_layout(shape || inferred, flat.length)
      @values = flat.map { |value| Value.wrap(value) }
      @storage = Storage.new(nil, @values)
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

      scalar_at(indices.each_with_index.sum { |index, axis| index * @logical_strides[axis] })
    end
    alias [] at

    def to_a
      data = flat_data
      build = lambda do |axis, offset|
        return data[offset] if axis == rank

        Array.new(shape[axis]) { |i| build.call(axis + 1, offset + i * @logical_strides[axis]) }
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

    def sum(axis: nil, keepdim: false)
      reduce_sum(axis, keepdim, average: false)
    end

    def mean(axis: nil, keepdim: false)
      reduce_sum(axis, keepdim, average: true)
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

    def contiguous?
      @contiguous
    end

    def contiguous
      return self if contiguous?

      operation(shape, flat_data, [@node]) do |gradient, pass|
        @node.accumulate_vector(pass, gradient.dup)
      end
    end

    def reshape(*dimensions)
      dimensions = dimensions.first.dup if dimensions.size == 1 && dimensions.first.is_a?(Array)
      unless dimensions.count(-1) <= 1 && dimensions.all? { |d| d.is_a?(Integer) && (d.positive? || d == -1) }
        raise ArgumentError, 'expected positive dimensions and at most one inferred dimension'
      end
      if (inferred = dimensions.index(-1))
        known = dimensions.reject { |d| d == -1 }.inject(1, :*)
        raise ArgumentError, 'cannot infer reshape dimension' unless (size % known).zero?

        dimensions[inferred] = size / known
      end
      raise ArgumentError, 'reshape must preserve element count' unless dimensions.inject(1, :*) == size

      contiguous.view(dimensions, nil) do |gradient, pass, source|
        source.accumulate_vector(pass, gradient.dup)
      end
    end

    def permute(*axes)
      axes = axes.first if axes.size == 1 && axes.first.is_a?(Array)
      unless axes.size == rank && axes.all? { |axis| axis.is_a?(Integer) } && axes.sort == (0...rank).to_a
        raise ArgumentError, 'axes must be a permutation of 0...rank'
      end

      axes = axes.dup.freeze
      dimensions = axes.map { |axis| shape[axis] }
      view(dimensions, axes.map { |axis| strides[axis] }) do |gradient, pass, source|
        gradient.each_with_index do |value, index|
          coordinates = coordinates_for(index, dimensions)
          original = coordinates.each_with_index.sum { |coordinate, axis| coordinate * @logical_strides[axes[axis]] }
          source.accumulate(pass, original, value)
        end
      end
    end

    def transpose
      raise ArgumentError, 'transpose requires a rank-2 tensor' unless rank == 2

      permute(1, 0)
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

    # Multiply the last two axes and broadcast the leading batch axes.
    def matmul(other)
      other = Tensor.wrap(other)
      raise ArgumentError, 'matmul requires rank-2 or higher tensors' unless rank >= 2 && other.rank >= 2

      rows, inner = shape.last(2)
      right_inner, columns = other.shape.last(2)
      raise ArgumentError, 'matmul dimension mismatch' unless inner == right_inner

      batch_shape = Tensor.broadcast_shape(shape[0...-2], other.shape[0...-2])
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      if batch_shape.empty?
        output = CPU.matmul(left, right, rows, inner, columns)
        return operation([rows, columns], output, [@node, right_node]) do |gradient, pass|
          grad_left, grad_right = CPU.matmul_backward(left, right, gradient, rows, inner, columns)
          @node.accumulate_vector(pass, grad_left)
          right_node.accumulate_vector(pass, grad_right)
        end
      end

      left_size = rows * inner
      right_size = inner * columns
      output_size = rows * columns
      left_matrices, right_matrices = {}, {}
      batches = Array.new(batch_shape.inject(1, :*)) do |index|
        coordinates = coordinates_for(index, batch_shape) + [0, 0]
        a, b = broadcast_offset(coordinates), other.broadcast_offset(coordinates)
        [a, b, left_matrices[a] ||= left.slice(a, left_size), right_matrices[b] ||= right.slice(b, right_size)]
      end
      output = batches.flat_map do |_, _, left_matrix, right_matrix|
        CPU.matmul(left_matrix, right_matrix, rows, inner, columns)
      end
      operation(batch_shape + [rows, columns], output, [@node, right_node]) do |gradient, pass|
        grad_left = Array.new(left.size, 0.0)
        grad_right = Array.new(right.size, 0.0)
        batches.each_with_index do |(a, b, left_matrix, right_matrix), index|
          upstream = gradient.slice(index * output_size, output_size)
          da, db = CPU.matmul_backward(left_matrix, right_matrix, upstream, rows, inner, columns)
          da.each_with_index { |value, i| grad_left[a + i] += value }
          db.each_with_index { |value, i| grad_right[b + i] += value }
        end
        @node.accumulate_vector(pass, grad_left)
        right_node.accumulate_vector(pass, grad_right)
      end
    end

    protected

    def graph_node; @node; end

    def flat_data
      Array.new(size) { |index| @storage.read(storage_offset(index)) }
    end

    def broadcast_offset(coordinates)
      coordinates.last(rank).each_with_index.sum do |coordinate, axis|
        shape[axis] == 1 ? 0 : coordinate * @logical_strides[axis]
      end
    end

    private

    def reduce_sum(axis, keepdim, average:)
      data = flat_data
      if axis.nil? && !keepdim
        divisor = average ? size : 1
        return reduce_with_gradient(data.sum.fdiv(divisor), Array.new(size, 1.0 / divisor))
      end

      axes = axis.nil? ? (0...rank).to_a : (axis.is_a?(Array) ? axis : [axis])
      if (!axis.nil? && axes.empty?) ||
         !axes.all? { |index| index.is_a?(Integer) && index >= -rank && index < rank }
        raise ArgumentError, 'expected valid reduction axes'
      end
      axes = axes.map { |index| index.negative? ? index + rank : index }
      raise ArgumentError, 'duplicate reduction axes' unless axes.uniq == axes

      kept = (0...rank).to_a - axes
      dimensions = keepdim ? shape.each_index.map { |i| axes.include?(i) ? 1 : shape[i] } : shape.values_at(*kept)
      output = Array.new(dimensions.inject(1, :*), 0.0)
      divisor = average ? shape.values_at(*axes).inject(1, :*) : 1
      offsets = data.each_index.map do |index|
        coordinates = coordinates_for(index, shape)
        offset = kept.reduce(0) { |flat, i| flat * shape[i] + coordinates[i] }
        output[offset] += data[index]
        offset
      end
      output.map! { |value| value.fdiv(divisor) } if average
      operation(dimensions, output, [@node]) do |gradient, pass|
        @node.accumulate_vector(pass, offsets.map { |offset| gradient[offset].fdiv(divisor) })
      end
    end

    def scalar_bridge(value, gradient)
      weights = gradient.dup.freeze
      Value.new(value, parents: [@node], tensor_bridge: true) do |upstream, pass|
        weights.each_with_index { |weight, i| @node.accumulate(pass, i, upstream * weight) }
      end
    end

    def coordinates_for(index, dimensions)
      dimensions.reverse.map do |dimension|
        index, coordinate = index.divmod(dimension)
        coordinate
      end.reverse
    end

    def elementwise(other, operator)
      other = Tensor.wrap(other)
      result_shape = Tensor.broadcast_shape(shape, other.shape)
      left, right = flat_data, other.flat_data
      right_node = other.graph_node
      offsets = Array.new(result_shape.inject(1, :*)) do |index|
        coordinates = coordinates_for(index, result_shape)
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
      @logical_strides = @shape.reverse.map do |dimension|
        current = stride
        stride *= dimension
        current
      end.reverse.freeze
      @strides = @logical_strides
      @contiguous = true
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
      @values = Array.new(data.length)
      @storage = Storage.new(data, @values)
      initialize_result_graph(parents, &pullback)
    end

    def initialize_result_graph(parents, &pullback)
      @node = Node.new(size, parents) do |gradient, pass|
        @last_gradient = gradient
        @values.each_with_index { |value, i| value.grad = gradient[i] if value }
        pullback.call(gradient, pass) if pullback
      end
    end

    def storage_offset(index)
      return index unless @view && !contiguous?

      coordinates_for(index, shape).each_with_index.sum { |coordinate, axis| coordinate * strides[axis] }
    end

    protected def view(dimensions, strides, &pullback)
      result = Tensor.allocate
      result.initialize_view(dimensions, strides, size, @storage, @node, &pullback)
      result
    end

    protected def initialize_view(dimensions, strides, size, storage, source, &pullback)
      set_layout(dimensions, size)
      @strides = strides.freeze if strides
      @storage = storage
      @view = true
      @values = Array.new(size)
      @contiguous = shape.each_index.all? { |axis| shape[axis] == 1 || @strides[axis] == @logical_strides[axis] }
      if TensorRuby.grad_enabled?
        initialize_result_graph([source]) { |gradient, pass| pullback.call(gradient, pass, source) }
      else
        initialize_result_graph([])
      end
    end

    def scalar_at(index)
      @values[index] ||= begin
        bridge = ->(gradient, pass) { @node.accumulate(pass, index, gradient) }
        value = if @view
                  ViewValue.new(@storage, storage_offset(index), parents: [@node], tensor_bridge: true, &bridge)
                else
                  Value.new(@storage.read(index), parents: [@node], tensor_bridge: true, &bridge)
                end
        value.grad = @last_gradient[index] if @last_gradient
        value
      end
    end
  end
end
