#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  class INeuralNetwork {
   public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual neuro_layer_t feedforward(const neuro_layer_t& inputs) const = 0;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual void addLayers(std::vector<std::unique_ptr<ILayer>>&&) = 0;
    virtual void addLayer(std::unique_ptr<ILayer>) = 0;
    virtual void addLayer(const ILayer*) = 0;
    virtual void addLayer(std::function<std::unique_ptr<ILayer>()> factory, size_t size) = 0;
    virtual void addLayer(std::function<std::unique_ptr<ILayer>()> factory) = 0;

    virtual void reset() = 0;
    virtual void reset(const std::vector<int>& newStructure) = 0;

    virtual void clearLayers() = 0;

    virtual void removeLayer(size_t index) = 0;
    virtual void popLayer() = 0;
    virtual void shiftLayer() = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual size_t sizeLayers() const = 0;

    virtual void setLayers(std::vector<std::unique_ptr<ILayer>>) = 0;

    virtual void setLayer(size_t index, std::unique_ptr<ILayer>) = 0;

    virtual const std::vector<std::unique_ptr<ILayer>>& getLayers() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>& getLayers() = 0;

    virtual std::vector<layer_weight_t> getAllWeights() const = 0;
    virtual std::vector<layer_bias_t> getAllBiases() const = 0;

    virtual const RefProxy<ILayer> layer(size_t index) const = 0;
    virtual RefProxy<ILayer> layer(size_t index) = 0;

    virtual std::unique_ptr<INeuralNetwork> clone() const = 0;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator begin() = 0;

    virtual std::vector<std::unique_ptr<ILayer>>::const_iterator end() const = 0;
    virtual std::vector<std::unique_ptr<ILayer>>::iterator end() = 0;

    virtual neuro_layer_t operator()(const neuro_layer_t& inputs) const = 0;

    virtual const ILayer& operator[](size_t index) const = 0;
    virtual ILayer& operator[](size_t index) = 0;
  };

}; // namespace neuro
