#pragma once

#include "neuro/interfaces/neural_network/i_neural_network.hpp"
#include "neuro/utils/interfaces/i_crossable.hpp"

namespace neuro {

class IGeneticTrainable : ICrossable<INeuralNetwork> {
 public:
  virtual ~IGeneticTrainable() = default;

  virtual void mutate(float rate, float intensity) = 0;
};

};  // namespace neuro
