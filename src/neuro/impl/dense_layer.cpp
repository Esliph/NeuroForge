#include "neuro/impl/dense_layer.hpp"

#include <memory>
#include <random>
#include <vector>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/activation.hpp"
#include "neuro/utils/random_engine.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  DenseLayer::DenseLayer(int inputSize, int outputSize)
      : ILayer(),
        weights(outputSize, neuro_layer_t(inputSize)),
        biases(outputSize) {}

  DenseLayer::DenseLayer(int inputSize, int outputSize, const ActivationFunction& activation)
      : ILayer(),
        weights(outputSize, neuro_layer_t(inputSize)),
        biases(outputSize),
        activation(activation) {}

  DenseLayer::DenseLayer(const ActivationFunction& activation)
      : ILayer(),
        activation(activation) {}

  DenseLayer::DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, const ActivationFunction& activation)
      : ILayer(),
        weights(weights),
        biases(biases),
        activation(activation) {}

  DenseLayer::DenseLayer(const layer_weight_t& weights, const ActivationFunction& activation)
      : ILayer(),
        weights(weights),
        biases(weights.size()),
        activation(activation) {}

  neuro_layer_t DenseLayer::feedforward(const neuro_layer_t& inputs) const {
    neuro_layer_t outputs(biases.size());

    for (size_t i = 0; i < outputs.size(); i++) {
      float total = biases[i];

      for (size_t j = 0; j < inputs.size(); j++) {
        total += weights[i][j] * inputs[j];
      }

      outputs[i] = activation.activate(total);
    }

    return outputs;
  }

  void DenseLayer::reset() {
    size_t outputLength = outputSize();

    weights = layer_weight_t(outputLength, neuro_layer_t(inputSize()));
    biases = layer_bias_t(outputLength);
  }

  void DenseLayer::randomizeWeights(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);

    for (auto& neurons : weights) {
      for (auto& weight : neurons) {
        weight = dist(random_engine);
      }
    }
  }

  void DenseLayer::randomizeBiases(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);

    for (auto& bias : biases) {
      bias = dist(random_engine);
    }
  }

  size_t DenseLayer::inputSize() const {
    return weights.empty() ? 0 : weights[0].size();
  }

  size_t DenseLayer::outputSize() const {
    return weights.size();
  }

  RefProxy<float> DenseLayer::weight(int indexX, int indexY) {
    return RefProxy<float>(weights[indexX][indexY]);
  }

  RefProxy<float> DenseLayer::bias(int index) {
    return RefProxy<float>(biases[index]);
  }

  void DenseLayer::setActivationFunction(const ActivationFunction& activation) {
    this->activation = activation;
  }

  void DenseLayer::setWeights(const layer_weight_t& weights) {
    this->weights = weights;
  }

  void DenseLayer::setBiases(const layer_bias_t& biases) {
    this->biases = biases;
  }

  void DenseLayer::setWeight(int indexX, int indexY, float value) {
    weights[indexX][indexY] = value;
  }

  void DenseLayer::setBias(int index, float value) {
    biases[index] = value;
  }

  const layer_weight_t& DenseLayer::getWeights() const {
    return weights;
  }

  const layer_bias_t& DenseLayer::getBiases() const {
    return biases;
  }

  layer_weight_t& DenseLayer::getWeights() {
    return weights;
  }

  layer_bias_t& DenseLayer::getBiases() {
    return biases;
  }

  const ActivationFunction& DenseLayer::getActivationFunction() const {
    return activation;
  }

  ILayer& DenseLayer::operator=(const ILayer& other) {
    activation = other.getActivationFunction();
    biases = other.getBiases();
    weights = other.getWeights();

    return *this;
  }

  std::unique_ptr<ILayer> DenseLayer::clone() const {
    return std::make_unique<DenseLayer>(*this);
  }

};  // namespace neuro
