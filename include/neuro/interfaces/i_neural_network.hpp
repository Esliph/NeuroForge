#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "neuro/capabilities/i_neural_network_iterable.hpp"
#include "neuro/capabilities/i_neural_network_runner.hpp"
#include "neuro/capabilities/i_neural_network_structure.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/types.hpp"

namespace neuro {

  class INeuralNetwork
    : public INeuralNetworkRunner,
      public INeuralNetworkStructure,
      public INeuralNetworkIterable {
   public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual std::unique_ptr<INeuralNetwork> clone() const = 0;
  };

}; // namespace neuro
