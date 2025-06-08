#pragma once

#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

class IGeneticTrainable {
 public:
  IGeneticTrainable() = default;
  virtual ~IGeneticTrainable() = default;

  virtual void mutate(float rate, float intensity) = 0;

  virtual INeuralNetwork crossover(const INeuralNetwork& partner) const = 0;
};

};  // namespace neuro
