#pragma once

#include <functional>

#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

class IGeneticPopulation {
 public:
  IGeneticPopulation(const IGeneticPopulation&) = default;
  virtual ~IGeneticPopulation() = default;

  virtual void evaluate(const std::function<float(const INeuralNetwork&)>& evaluateFunction) = 0;

  virtual void mutate(float rate, float intensity) = 0;
  virtual void crossover() = 0;
};

};  // namespace neuro
