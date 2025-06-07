#pragma once

#include "neuro/interfaces/layer/i_layer.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

class DenseLayer : public ILayer {
  layer_weight_t weights{};
  layer_bias_t biases{};

  ActivationFunction activation;

 public:
  DenseLayer() = default;
  DenseLayer(const DenseLayer&) = default;

  DenseLayer(int inputSize, int outputSize, ActivationFunction& activation);
  DenseLayer(const layer_weight_t& weights, ActivationFunction& activation);
  DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, ActivationFunction& activation);

  virtual ~DenseLayer() = default;

  neuro_layer_t feedforward(const neuro_layer_t& inputs) override;

  void randomizeWeights(float min, float max) override;
  void randomizeBiases(float min, float max) override;

  void setActivationFunction(const ActivationFunction&) override;

  void setWeights(const layer_weight_t&) override;
  void setBiases(const layer_bias_t&) override;

  void setWeight(int indexX, int indexY, float value) override;
  void setBias(int index, float value) override;

  const layer_weight_t& getWeights() const override;
  const layer_bias_t& getBiases() const override;

  layer_weight_t& getWeights() override;
  layer_bias_t& getBiases() override;

  const ActivationFunction& getActivationFunction() const override;

  float getWeight(int indexX, int indexY) const override;
  float getBias(int index) const override;

  size_t inputSize() const override;
  size_t outputSize() const override;
};

};  // namespace neuro
