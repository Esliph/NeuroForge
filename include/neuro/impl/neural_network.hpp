#pragma once

#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

class NeuralNetwork : INeuralNetwork {
 public:
  NeuralNetwork(NeuralNetwork&) = default;
  virtual ~NeuralNetwork() = default;
};

};  // namespace neuro
