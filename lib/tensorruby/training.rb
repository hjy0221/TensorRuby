module TensorRuby
  module Training
    # An epoch consumes [input, target] pairs, where each pair may itself be a batch.
    class Trainer
      attr_reader :steps

      def initialize(model, optimizer:, loss:, weight: nil, max_grad_norm: nil, forward: nil)
        @model = model
        @optimizer = optimizer
        @loss = loss
        @weight = weight || ->(_target) { 1 }
        @forward = forward || model.method(:call)
        @max_grad_norm = max_grad_norm
        @parameters = model.parameters if max_grad_norm
        @steps = 0
      end

      # A callable learning rate receives the zero-based optimizer step.
      def train_epoch(examples, accumulation_steps: 1, learning_rate: nil)
        unless accumulation_steps.is_a?(Integer) && accumulation_steps.positive?
          raise ArgumentError, 'accumulation_steps must be a positive integer'
        end
        raise ArgumentError, 'training requires gradients' unless TensorRuby.grad_enabled?

        @model.train
        total_loss = 0.0
        total_weight = 0.0
        examples.each_slice(accumulation_steps) do |batch|
          weighted = batch.map { |input, target| [input, target, weight_for(target)] }
          batch_weight = weighted.sum(&:last)
          next if batch_weight.zero?

          @optimizer.zero_grad
          weighted.each do |input, target, weight|
            next if weight.zero?

            loss = loss_for(input, target)
            loss.backward(weight.fdiv(batch_weight))
            total_loss += loss.data * weight
          end
          Optim.clip_grad_norm(@parameters, @max_grad_norm) if @max_grad_norm
          if learning_rate
            @optimizer.step(lr: learning_rate.call(@steps))
          else
            @optimizer.step
          end
          @steps += 1
          total_weight += batch_weight
        end
        mean_loss(total_loss, total_weight)
      end

      def evaluate(examples)
        training = @model.training?
        @model.eval
        total_loss = 0.0
        total_weight = 0.0
        TensorRuby.no_grad do
          examples.each do |input, target|
            weight = weight_for(target)
            next if weight.zero?

            total_loss += loss_for(input, target).data * weight
            total_weight += weight
          end
        end
        mean_loss(total_loss, total_weight)
      ensure
        @model.train(training)
      end

      private

      def weight_for(target)
        weight = @weight.call(target)
        unless weight.is_a?(Numeric) && weight.real? && weight.finite? && weight >= 0
          raise ArgumentError, 'example weight must be finite and nonnegative'
        end

        weight
      end

      def loss_for(input, target)
        loss = @loss.call(@forward.call(input), target)
        raise ArgumentError, 'loss must be a finite scalar Value' unless loss.is_a?(Value) && loss.data.finite?

        loss
      end

      def mean_loss(total, weight)
        raise ArgumentError, 'at least one example must have positive weight' if weight.zero?

        total.fdiv(weight)
      end
    end
  end
end
