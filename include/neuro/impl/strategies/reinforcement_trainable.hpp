#pragma once

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/strategies/i_reinforcement_trainable.hpp"

namespace neuro {

class IReinforcementNeuralNetwork : public INeuralNetwork, public IReinforcementNeuralNetwork {
 public:
  IReinforcementNeuralNetwork() = default;
  virtual ~IReinforcementNeuralNetwork() = default;
};

};  // namespace neuro
