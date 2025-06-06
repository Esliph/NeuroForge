#pragma once

#include "neuro/interfaces/neural_network/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_reinforcement_trainable.hpp"

namespace neuro {

class IReinforcementNeuralNetwork : INeuralNetwork, IReinforcementNeuralNetwork {
 public:
  virtual ~IReinforcementNeuralNetwork() = default;
};

};  // namespace neuro
