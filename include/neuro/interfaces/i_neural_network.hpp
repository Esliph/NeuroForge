#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/layer/i_layer.hpp"

namespace neuro {

class INeuralNetwork {
 public:
  INeuralNetwork() = default;
  INeuralNetwork(const INeuralNetwork&) = default;
  virtual ~INeuralNetwork() = default;

  virtual void addLayer(std::unique_ptr<ILayer>) = 0;

  virtual void clearLayers() = 0;

  virtual void randomizeAll(float max = 1.0f) {
    randomizeAll(-max, max, -max, max);
  }

  virtual void randomizeAll(float min, float max) {
    randomizeAll(min, max, min, max);
  }

  virtual void randomizeAll(float minWeight, float maxWeight, float minBias, float maxBias) {
    randomizeWeights(minWeight, maxWeight);
    randomizeBiases(minBias, maxBias);
  }

  virtual void randomizeWeights(float max = 1.0f) {
    randomizeWeights(-max, max);
  }

  virtual void randomizeBiases(float max = 1.0f) {
    randomizeBiases(-max, max);
  }

  virtual void randomizeWeights(float min, float max) = 0;
  virtual void randomizeBiases(float min, float max) = 0;

  virtual void setAllWeights(const std::vector<layer_weight_t>&) = 0;
  virtual void setAllBiases(const std::vector<layer_bias_t>&) = 0;

  virtual void setLayers(const std::vector<std::unique_ptr<ILayer>>&) = 0;

  virtual const ILayer& getLayer(size_t index) const = 0;
  virtual size_t getNumLayers() const = 0;

  virtual INeuralNetwork& operator=(const INeuralNetwork&) = default;
};

};  // namespace neuro
