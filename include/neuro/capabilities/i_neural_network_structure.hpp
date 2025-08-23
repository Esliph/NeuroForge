#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"

namespace neuro {

  class INeuralNetworkStructure {
   public:
    virtual ~INeuralNetworkStructure() = default;

    virtual void addLayer(std::unique_ptr<ILayer>) = 0;
    virtual void addLayers(std::vector<std::unique_ptr<ILayer>>&&) = 0;
    virtual void addLayer(std::function<std::unique_ptr<ILayer>()> factory) = 0;
    virtual void addLayer(std::function<std::unique_ptr<ILayer>()> factory, size_t size) = 0;

    virtual void removeLayer(size_t index) = 0;
    virtual void popLayer() = 0;
    virtual void shiftLayer() = 0;

    virtual void clearLayers() = 0;
    virtual void clear() = 0;
    virtual void restructure(const std::vector<int>& newStructure) = 0;

    virtual size_t sizeLayers() const = 0;

    virtual void setLayers(std::vector<std::unique_ptr<ILayer>>) = 0;
    virtual void setLayer(size_t index, std::unique_ptr<ILayer>) = 0;
  };

}; // namespace neuro
