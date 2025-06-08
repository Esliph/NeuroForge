#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/layer/i_layer.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

class NeuralNetwork : public INeuralNetwork {
  std::vector<std::unique_ptr<ILayer>> layers{};

 public:
  NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork&) = default;

  NeuralNetwork(const std::vector<std::unique_ptr<ILayer>>& layers);
  NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation);
  NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

  virtual ~NeuralNetwork() = default;

  void addLayer(std::unique_ptr<ILayer>) override;

  void clearLayers() override;
  virtual std::vector<std::unique_ptr<ILayer>> clearLayersAndReturn() = 0;

  void randomizeAll(float minWeight, float maxWeight, float minBias, float maxBias) override;

  void randomizeWeights(float min, float max) override;
  void randomizeBiases(float min, float max) override;

  void setAllWeights(const std::vector<layer_weight_t>&) override;
  void setAllBiases(const std::vector<layer_bias_t>&) override;

  void setLayers(const std::vector<std::unique_ptr<ILayer>>&) override;

  const ILayer& getLayer(size_t index) const override;
  size_t getNumLayers() const override;
};

};  // namespace neuro
