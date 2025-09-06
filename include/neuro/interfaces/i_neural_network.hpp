#pragma once

#include "neuro/capabilities/i_neural_network_iterable.hpp"
#include "neuro/capabilities/i_neural_network_parameter.hpp"
#include "neuro/capabilities/i_neural_network_runner.hpp"
#include "neuro/capabilities/i_neural_network_structure.hpp"

namespace neuro {

  class INeuralNetwork
    : public INeuralNetworkRunner,
      public INeuralNetworkStructure,
      public INeuralNetworkIterable,
      public INeuralNetworkParameter {
   public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual std::unique_ptr<INeuralNetwork> clone() const = 0;
  };

} // namespace neuro
