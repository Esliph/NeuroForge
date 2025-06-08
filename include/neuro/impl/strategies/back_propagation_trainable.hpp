#pragma once

#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/strategies/i_back_propagation_trainable.hpp"

namespace neuro {

class BackPropagationNeuralNetwork : public NeuralNetwork, public IBackPropagationTrainable {
 public:
  BackPropagationNeuralNetwork() = default;
  virtual ~BackPropagationNeuralNetwork() = default;
};

};  // namespace neuro
