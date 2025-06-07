#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_back_propagation_trainable.hpp"

namespace neuro {

class BackPropagationNeuralNetwork : INeuralNetwork, IBackPropagationTrainable {
 public:
  virtual ~BackPropagationNeuralNetwork() = default;
};

};  // namespace neuro
