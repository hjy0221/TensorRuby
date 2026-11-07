require_relative 'lib/rubygrad/version'
Gem::Specification.new do |s|
  s.name = 'rubygrad'; s.version = Rubygrad::VERSION
  s.summary = 'Pure Ruby automatic differentiation and neural networks'
  s.authors = ['Rubygrad contributors']; s.license = 'MIT'
  s.files = Dir['lib/**/*.rb', 'README.md']; s.require_paths = ['lib']
  s.required_ruby_version = '>= 2.7'
end
