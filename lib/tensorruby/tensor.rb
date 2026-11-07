module TensorRuby
  # Row-major contiguous scalar-backed tensor.
  class Tensor
    attr_reader :shape, :strides
    def initialize(data, shape: nil)
      inferred, flat = self.class.flatten_data(data)
      @shape = (shape || inferred).dup.freeze
      raise ArgumentError, 'dimensions must be positive integers' unless @shape.all? { |d| d.is_a?(Integer) && d.positive? }
      raise ArgumentError, 'shape does not match data' unless @shape.inject(1, :*) == flat.length
      @strides = @shape.each_index.map { |i| @shape[(i + 1)..-1].inject(1, :*) }.freeze
      @values = flat.map { |v| Value.wrap(v) }.freeze
    end
    def self.flatten_data(data)
      return [[], [data]] unless data.is_a?(Array)
      raise ArgumentError, 'empty dimensions unsupported' if data.empty?
      parts = data.map { |v| flatten_data(v) }
      raise ArgumentError, 'ragged tensor' unless parts.all? { |p| p[0] == parts[0][0] }
      [[data.length] + parts[0][0], parts.flat_map(&:last)]
    end
    def self.wrap(x); x.is_a?(Tensor) ? x : new(x); end
    def values; @values.dup; end
    def size; @values.length; end
    def rank; shape.length; end
    def at(*indices)
      raise ArgumentError, 'incorrect index count' unless indices.length == rank
      raise IndexError, 'index out of bounds' unless indices.each_with_index.all? { |x, i| x.is_a?(Integer) && x >= 0 && x < shape[i] }
      @values[indices.each_with_index.sum { |x, i| x * strides[i] }]
    end
    alias [] at
    def to_a
      build = lambda { |axis, offset| axis == rank ? @values[offset].data : Array.new(shape[axis]) { |i| build.call(axis + 1, offset + i * strides[axis]) } }
      build.call(0, 0)
    end
    def self.broadcast_shape(a, b)
      n = [a.length, b.length].max
      ([1] * (n - a.length) + a).zip([1] * (n - b.length) + b).map do |x, y|
        raise ArgumentError, 'incompatible broadcast shapes' unless x == y || x == 1 || y == 1
        [x, y].max
      end
    end
    def self.coordinates(shape)
      return [[]] if shape.empty?
      coordinates(shape[0...-1]).flat_map { |prefix| (0...shape[-1]).map { |i| prefix + [i] } }
    end
    def broadcast_value(coords)
      at(*coords.last(rank).each_with_index.map { |v, i| shape[i] == 1 ? 0 : v })
    end
    def elementwise(other, op)
      other = Tensor.wrap(other); result_shape = Tensor.broadcast_shape(shape, other.shape)
      vals = Tensor.coordinates(result_shape).map { |idx| broadcast_value(idx).public_send(op, other.broadcast_value(idx)) }
      Tensor.new(vals, shape: result_shape)
    end
    def +(other); elementwise(other, :+); end
    def -(other); elementwise(other, :-); end
    def *(other); elementwise(other, :*); end
    def /(other); elementwise(other, :/); end
    def coerce(other); [Tensor.wrap(other), self]; end
    def relu; Tensor.new(@values.map(&:relu), shape: shape); end
    def sum; @values.reduce(:+); end
    def mean; sum / size; end
    def matmul(other)
      other = Tensor.wrap(other)
      raise ArgumentError, 'matmul requires rank-2 tensors' unless rank == 2 && other.rank == 2
      raise ArgumentError, 'matmul dimension mismatch' unless shape[1] == other.shape[0]
      rows, inner, cols = shape[0], shape[1], other.shape[1]
      vals = (0...rows).flat_map { |i| (0...cols).map { |j| (0...inner).map { |k| at(i, k) * other.at(k, j) }.reduce(:+) } }
      Tensor.new(vals, shape: [rows, cols])
    end
  end
end
