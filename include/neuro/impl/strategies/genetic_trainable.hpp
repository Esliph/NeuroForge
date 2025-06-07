#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_genetic_trainable.hpp"

namespace neuro {

class GeneticNeuralNetwork : public INeuralNetwork, public IGeneticTrainable {
 public:
  GeneticNeuralNetwork() = default;
  virtual ~GeneticNeuralNetwork() = default;
};

};  // namespace neuro
