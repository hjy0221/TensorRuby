require_relative 'lib/tensorruby/version'
Gem::Specification.new do |s|
  s.name = 'tensorruby'; s.version = TensorRuby::VERSION
  s.homepage = 'https://github.com/hjy0221/TensorRuby'
  s.summary = 'Pure Ruby automatic differentiation and neural networks'
  s.authors = ['TensorRuby contributors']; s.license = 'MIT'
  s.files = Dir['lib/**/*.rb', 'README*.md']; s.require_paths = ['lib']
  s.required_ruby_version = '>= 2.6'
end
