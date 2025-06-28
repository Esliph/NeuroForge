#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/common/i_neural_operable.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  class INeuralNetwork : public INeuralOperable {
   public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual void addLayer(std::unique_ptr<ILayer>) = 0;

    virtual void reset() = 0;

    virtual void clearLayers() = 0;

    virtual void removeLayer(size_t index) = 0;
    virtual void popLayer() = 0;

    virtual const std::vector<std::unique_ptr<ILayer>>& getLayers() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>& getLayers() = 0;

    virtual const RefProxy<ILayer> layer(size_t index) const = 0;
    virtual RefProxy<ILayer> layer(size_t index) = 0;

    virtual std::unique_ptr<INeuralNetwork> clone() const = 0;
  };

};  // namespace neuro
