#pragma once

#include <memory>

#include "neuro/capabilities/i_layer_operation.hpp"
#include "neuro/capabilities/i_layer_structure.hpp"
#include "neuro/capabilities/i_layer_weight.hpp"

namespace neuro {

  class ILayer
    : public ILayerOperation,
      public ILayerStructure,
      public ILayerWeight {
   public:
    ILayer() = default;
    virtual ~ILayer() = default;

    virtual std::unique_ptr<ILayer> clone() const = 0;

    ILayer& operator=(const ILayer&) = default;
  };

}; // namespace neuro
