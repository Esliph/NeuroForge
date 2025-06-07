#pragma once

#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

class NeuralNetwork : public INeuralNetwork {
 public:
  NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork&) = default;
  virtual ~NeuralNetwork() = default;
};

};  // namespace neuro
