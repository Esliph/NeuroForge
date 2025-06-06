#pragma once

#include "neuro/interfaces/neural_network/i_neural_network.hpp"

namespace neuro {

class NeuralNetwork : INeuralNetwork {
 public:
  NeuralNetwork(NeuralNetwork&) = default;
  virtual ~NeuralNetwork() = default;
};

};  // namespace neuro