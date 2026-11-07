module TensorRuby
  module Data
    # Groups indexable [input, target] samples into tensor mini-batches.
    class DataLoader
      include Enumerable

      def initialize(dataset, batch_size:, shuffle: true, random: Random.new)
        unless batch_size.is_a?(Integer) && batch_size.positive?
          raise ArgumentError, 'batch_size must be a positive integer'
        end

        @dataset = dataset
        @batch_size = batch_size
        @shuffle = shuffle
        @random = random
      end

      def each
        return enum_for(:each) { size } unless block_given?

        indices = (0...@dataset.size).to_a
        indices.shuffle!(random: @random) if @shuffle
        indices.each_slice(@batch_size) do |batch|
          inputs, targets = batch.map { |index| @dataset[index] }.transpose
          yield Tensor.new(inputs), Tensor.new(targets)
        end
        self
      end

      def size
        (@dataset.size + @batch_size - 1) / @batch_size
      end
    end
  end
end
