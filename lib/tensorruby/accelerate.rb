module TensorRuby
  module Backend
    # Double-precision row-major GEMM through macOS Accelerate.
    module Accelerate
      def self.available?
        !gemm.nil?
      end

      def self.matmul(left, right, rows, inner, columns)
        multiply(left, right, rows, inner, columns)
      end

      def self.matmul_backward(left, right, gradient, rows, inner, columns)
        [
          multiply(gradient, right, rows, columns, inner, transpose_right: true),
          multiply(left, gradient, inner, rows, columns, transpose_left: true)
        ]
      end

      def self.multiply(left, right, rows, inner, columns, transpose_left: false, transpose_right: false)
        kernel = gemm
        raise NotImplementedError, 'macOS Accelerate is unavailable' unless kernel
        unless [rows, inner, columns].all? { |dimension| dimension.between?(1, 2**31 - 1) } &&
               left.size == rows * inner && right.size == inner * columns
          raise ArgumentError, 'invalid GEMM dimensions or buffer sizes'
        end

        output = "\0".b * (8 * rows * columns)
        kernel.call(101, transpose_left ? 112 : 111, transpose_right ? 112 : 111,
                    rows, columns, inner, 1.0, left.pack('d*'), transpose_left ? rows : inner,
                    right.pack('d*'), transpose_right ? inner : columns, 0.0, output, columns)
        output.unpack('d*')
      end

      def self.gemm
        return @gemm if defined?(@gemm)

        @gemm = if RUBY_PLATFORM.include?('darwin')
                  begin
                    require 'fiddle'
                  rescue LoadError
                    return @gemm = nil
                  end
                  @library = Fiddle.dlopen('/System/Library/Frameworks/Accelerate.framework/Accelerate')
                  int, double, pointer = Fiddle::TYPE_INT, Fiddle::TYPE_DOUBLE, Fiddle::TYPE_VOIDP
                  Fiddle::Function.new(@library['cblas_dgemm'],
                                       [int, int, int, int, int, int, double, pointer, int,
                                        pointer, int, double, pointer, int], Fiddle::TYPE_VOID)
                end
      rescue Fiddle::DLError
        @gemm = nil
      end

      private_class_method :multiply, :gemm
    end
  end
end
