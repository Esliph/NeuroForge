#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/interfaces/i_crossable.hpp"

namespace neuro {

class IGeneticTrainable : ICrossable<INeuralNetwork> {
 public:
  IGeneticTrainable() = default;
  virtual ~IGeneticTrainable() = default;

  virtual void mutate(float rate, float intensity) = 0;
};

};  // namespace neuro
