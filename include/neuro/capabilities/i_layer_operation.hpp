#pragma once

#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class ILayerOperation {
   public:
    virtual ~ILayerOperation() = default;

    virtual neuro_layer_t feedforward(const neuro_layer_t& inputs) const = 0;

    virtual const ActivationFunction& getActivationFunction() const = 0;
    virtual void setActivationFunction(const ActivationFunction&) = 0;
  };

} // namespace neuro
