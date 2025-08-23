#pragma once

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/types.hpp"

namespace neuro {

  class INeuralNetworkRunner {
   public:
    virtual ~INeuralNetworkRunner() = default;

    virtual neuro_layer_t feedforward(const neuro_layer_t& inputs) const = 0;
    virtual neuro_layer_t operator()(const neuro_layer_t& inputs) const = 0;

    virtual const ILayer& layer(size_t index) const = 0;
    virtual ILayer& layer(size_t index) = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual bool empty() const = 0;
  };

}; // namespace neuro
