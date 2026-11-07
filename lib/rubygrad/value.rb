module Rubygrad
  class Value
    attr_accessor :data, :grad
    attr_reader :parents
    def initialize(data, parents: [], &backward)
      raise TypeError, 'data must be numeric' unless data.is_a?(Numeric)
      @data, @grad, @parents = data.to_f, 0.0, parents.freeze
      @backward = backward || ->(_g) {}
    end
    def self.wrap(x); x.is_a?(Value) ? x : new(x); end
    def +(other)
      other = Value.wrap(other)
      Value.new(data + other.data, parents: [self, other]) { |g| self.grad += g; other.grad += g }
    end
    def -@; self * -1; end
    def -(other); self + -Value.wrap(other); end
    def *(other)
      other = Value.wrap(other); a, b = data, other.data
      Value.new(a * b, parents: [self, other]) { |g| self.grad += b * g; other.grad += a * g }
    end
    def /(other); self * (Value.wrap(other)**-1); end
    def **(power)
      raise TypeError, 'power must be numeric' unless power.is_a?(Numeric)
      base = data
      Value.new(base**power, parents: [self]) { |g| self.grad += power * base**(power - 1) * g }
    end
    def relu
      active = data > 0
      Value.new(active ? data : 0, parents: [self]) { |g| self.grad += g if active }
    end
    def coerce(other); [Value.wrap(other), self]; end
    def zero_grad; @grad = 0.0; self; end
    # Clear intermediate gradients each pass, while accumulating leaf gradients.
    def backward(seed = 1.0)
      raise TypeError, 'seed must be numeric' unless seed.is_a?(Numeric)
      order, seen, stack = [], {}, [[self, false]]
      until stack.empty?
        node, expanded = stack.pop
        if expanded
          order << node
        elsif !seen[node.object_id]
          seen[node.object_id] = true
          stack << [node, true]
          node.parents.reverse_each { |p| stack << [p, false] unless seen[p.object_id] }
        end
      end
      saved = {}
      order.each { |n| saved[n.object_id] = n.grad if n.parents.empty?; n.grad = 0.0 }
      self.grad = seed.to_f
      order.reverse_each { |n| n.instance_variable_get(:@backward).call(n.grad) }
      order.each { |n| n.grad += saved[n.object_id] if saved.key?(n.object_id) }
      self
    end
  end
end
