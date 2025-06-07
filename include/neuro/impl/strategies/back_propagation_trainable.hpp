#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_back_propagation_trainable.hpp"

namespace neuro {

class BackPropagationNeuralNetwork : public INeuralNetwork, public IBackPropagationTrainable {
 public:
  BackPropagationNeuralNetwork() = default;
  virtual ~BackPropagationNeuralNetwork() = default;
};

};  // namespace neuro
