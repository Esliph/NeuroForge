#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_genetic_trainable.hpp"

namespace neuro {

class GeneticNeuralNetwork : INeuralNetwork, IGeneticTrainable {
 public:
  virtual ~GeneticNeuralNetwork() = default;
};

};  // namespace neuro
