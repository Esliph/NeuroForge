#pragma once

#include <memory>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
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

  neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

  void addLayer(std::unique_ptr<ILayer>) override;

  void reset() override;

  void clearLayers() override;

  void removeLayer(size_t index) override;
  void popLayer() override;

  void randomizeWeights(float min, float max) override;
  void randomizeBiases(float min, float max) override;

  size_t inputSize() const override;
  size_t outputSize() const override;

  void setAllWeights(const std::vector<layer_weight_t>&) override;
  void setAllBiases(const std::vector<layer_bias_t>&) override;

  void setLayers(std::vector<std::unique_ptr<ILayer>>) override;

  std::vector<layer_weight_t> getAllWeights() const override;
  std::vector<layer_bias_t> getAllBiases() const override;

  const std::vector<std::unique_ptr<ILayer>>& getLayers() const override;
  std::vector<std::unique_ptr<ILayer>>& getLayers() override;

  const RefProxy<ILayer> layer(size_t index) const override;
  RefProxy<ILayer> layer(size_t index) override;

  size_t sizeLayers() const override;

  std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const override;
  std::vector<std::unique_ptr<ILayer>>::iterator begin() override;

  std::vector<std::unique_ptr<ILayer>>::const_iterator end() const override;
  std::vector<std::unique_ptr<ILayer>>::iterator end() override;

  neuro_layer_t operator()(const neuro_layer_t& inputs) const override;

  const ILayer& operator[](int index) const override;
  ILayer& operator[](int index) override;

  std::unique_ptr<INeuralNetwork> clone() const override;
};

};  // namespace neuro
