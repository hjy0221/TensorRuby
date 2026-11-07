module TensorRuby
  module Diffusion
    # A linear DDPM schedule with zero-based timesteps.
    class Schedule
      attr_reader :steps

      def initialize(steps:, beta_start: 0.0001, beta_end: 0.02)
        unless steps.is_a?(Integer) && steps >= 2
          raise ArgumentError, 'steps must be an integer of at least two'
        end
        unless 0 < beta_start && beta_start <= beta_end && beta_end < 1
          raise ArgumentError, 'expected 0 < beta_start <= beta_end < 1'
        end

        @steps = steps
        @betas = Array.new(steps) { |i| beta_start + (beta_end - beta_start) * i.fdiv(steps - 1) }
        @alphas = @betas.map { |beta| 1.0 - beta }
        product = 1.0
        @cumulative = @alphas.map { |alpha| product *= alpha }
      end

      def add_noise(clean, noise, step)
        unless step.is_a?(Integer) && step >= 0 && step < steps
          raise ArgumentError, 'timestep out of range'
        end

        clean, noise = Tensor.wrap(clean), Tensor.wrap(noise)
        raise ArgumentError, 'noise shape must match clean data' unless clean.shape == noise.shape

        alpha_bar = @cumulative[step]
        clean * Math.sqrt(alpha_bar) + noise * Math.sqrt(1.0 - alpha_bar)
      end

      def self.normal(shape, random: Random.new)
        unless shape.all? { |dimension| dimension.is_a?(Integer) && dimension.positive? }
          raise ArgumentError, 'shape dimensions must be positive integers'
        end

        data = Array.new(shape.inject(1, :*)) do
          Math.sqrt(-2.0 * Math.log(1.0 - random.rand)) * Math.cos(2.0 * Math::PI * random.rand)
        end
        Tensor.new(data, shape: shape)
      end

      # The block receives the current noisy tensor and timestep and predicts noise.
      # The optional observer receives (remaining_steps, sample) from initial noise to zero.
      def sample(shape, random: Random.new, on_step: nil)
        raise ArgumentError, 'a noise predictor block is required' unless block_given?

        TensorRuby.no_grad do
          current = self.class.normal(shape, random: random)
          on_step.call(steps, current) if on_step
          (steps - 1).downto(0) do |step|
            noise = Tensor.wrap(yield(current, step))
            raise ArgumentError, 'predicted noise shape must match sample shape' unless noise.shape == current.shape

            beta = @betas[step]
            remaining_noise = 1.0 - @cumulative[step]
            current = (current - noise * (beta / Math.sqrt(remaining_noise))) / Math.sqrt(@alphas[step])
            if step.positive?
              variance = beta * (1.0 - @cumulative[step - 1]) / remaining_noise
              current += self.class.normal(shape, random: random) * Math.sqrt(variance)
            end
            on_step.call(step, current) if on_step
          end
          current
        end
      end
    end
  end
end
