#pragma once

#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/strategies/i_genetic_trainable.hpp"

namespace neuro {

class GeneticNeuralNetwork : public NeuralNetwork, public IGeneticTrainable {
 public:
  GeneticNeuralNetwork() = default;
  virtual ~GeneticNeuralNetwork() = default;
};

};  // namespace neuro
