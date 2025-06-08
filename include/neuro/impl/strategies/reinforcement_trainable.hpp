#pragma once

#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/strategies/i_reinforcement_trainable.hpp"

namespace neuro {

class ReinforcementNeuralNetwork : public NeuralNetwork, public IReinforcementTrainable {
 public:
  ReinforcementNeuralNetwork() = default;
  virtual ~ReinforcementNeuralNetwork() = default;
};

};  // namespace neuro
