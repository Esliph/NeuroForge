#include "neuro/impl/dense_layer.hpp"

#include <memory>
#include <random>
#include <vector>

#include "internal/random_engine.hpp"
#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  DenseLayer::DenseLayer(const layer_weight_t& weights)
    : ILayer(),
      weights(weights),
      biases(weights.size()) {}

  DenseLayer::DenseLayer(const layer_bias_t& biases)
    : ILayer(),
      weights(biases.size()),
      biases(biases) {}

  DenseLayer::DenseLayer(const ActivationFunction& activation)
    : ILayer(),
      activation(activation) {}

  DenseLayer::DenseLayer(size_t inputSize, size_t outputSize)
    : ILayer(),
      weights(outputSize, neuro_layer_t(inputSize)),
      biases(outputSize) {}

  DenseLayer::DenseLayer(size_t inputSize, size_t outputSize, const ActivationFunction& activation)
    : ILayer(),
      weights(outputSize, neuro_layer_t(inputSize)),
      biases(outputSize),
      activation(activation) {}

  DenseLayer::DenseLayer(const layer_weight_t& weights, const ActivationFunction& activation)
    : ILayer(),
      weights(weights),
      biases(weights.size()),
      activation(activation) {}

  DenseLayer::DenseLayer(const layer_bias_t& biases, const ActivationFunction& activation)
    : ILayer(),
      weights(biases.size()),
      biases(biases),
      activation(activation) {}

  DenseLayer::DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases)
    : ILayer(),
      weights(weights),
      biases(biases) {}

  DenseLayer::DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, const ActivationFunction& activation)
    : ILayer(),
      weights(weights),
      biases(biases),
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

  void DenseLayer::reshape(size_t newInputSize, size_t newOutputSize) {
    weights = layer_weight_t(newOutputSize, neuro_layer_t(newInputSize));
    biases = layer_bias_t(newOutputSize);
  }

  void DenseLayer::randomizeWeights(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);

    for (size_t i = 0; i < weights.size(); i++) {
      for (size_t j = 0; j < weights[i].size(); j++) {
        weights[i][j] = dist(random_engine);
      }
    }
  }

  void DenseLayer::randomizeBiases(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);

    for (size_t i = 0; i < biases.size(); i++) {
      biases[i] = dist(random_engine);
    }
  }

  void DenseLayer::mutateWeights(const std::function<float(float)>& mutator) {
    for (size_t i = 0; i < weights.size(); i++) {
      for (size_t j = 0; j < weights[i].size(); j++) {
        weights[i][j] = mutator(weights[i][j]);
      }
    }
  }

  void DenseLayer::mutateBiases(const std::function<float(float)>& mutator) {
    for (size_t i = 0; i < biases.size(); i++) {
      biases[i] = mutator(biases[i]);
    }
  }

  bool DenseLayer::validateInternalShape(const layer_weight_t& weights, const layer_bias_t& biases) {
    auto expectedOutput = outputSize();

    if (expectedOutput == 0 || weights.size() != expectedOutput || biases.size() != expectedOutput) {
      return false;
    }

    auto expectedInput = inputSize();

    for (size_t i = 0; i < weights.size(); i++) {
      if (weights[i].size() != expectedInput) {
        return false;
      }
    }

    return true;
  }

  float DenseLayer::meanWeight() const {
    if (weights.empty()) {
      return 0;
    }

    float total = 0;

    for (size_t i = 0; i < weights.size(); i++) {
      for (size_t j = 0; j < weights[i].size(); j++) {
        total += weights[i][j];
      }
    }

    return total / (weights.size() * weights[0].size());
  }

  float DenseLayer::meanBias() const {
    if (biases.empty()) {
      return 0;
    }

    float total = 0;

    for (size_t i = 0; i < biases.size(); i++) {
      total += biases[i];
    }

    return total / biases.size();
  }

  float& DenseLayer::weightRef(size_t indexX, size_t indexY) {
    checkWeightIndex(indexX, indexY);
    return weights[indexX][indexY];
  }

  float& DenseLayer::biasRef(size_t index) {
    checkBiasIndex(index);
    return biases[index];
  }

  float DenseLayer::getWeight(size_t indexX, size_t indexY) const {
    checkWeightIndex(indexX, indexY);
    return weights[indexX][indexY];
  }

  float DenseLayer::getBias(size_t index) const {
    checkBiasIndex(index);
    return biases[index];
  }

  const float& DenseLayer::weightRef(size_t indexX, size_t indexY) const {
    checkWeightIndex(indexX, indexY);
    return weights[indexX][indexY];
  }

  const float& DenseLayer::biasRef(size_t index) const {
    checkBiasIndex(index);
    return biases[index];
  }

  void DenseLayer::setWeight(size_t indexX, size_t indexY, float value) {
    checkWeightIndex(indexX, indexY);
    weights[indexX][indexY] = value;
  }

  void DenseLayer::setBias(size_t index, float value) {
    checkBiasIndex(index);
    biases[index] = value;
  }

  ILayer& DenseLayer::operator=(const ILayer& other) {
    activation = other.getActivationFunction();
    biases = other.getBiases();
    weights = other.getWeights();

    return *this;
  }

  void DenseLayer::checkWeightIndex(size_t indexX, size_t indexY) const {
    if (indexX >= weights.size()) {
      throw exception::InvalidNetworkArchitectureException("Index out of range of the neuron output weight vector");
    }
    if (indexY >= weights[indexX].size()) {
      throw exception::InvalidNetworkArchitectureException("Index out of range of the neuron input weight vector");
    }
  }

  void DenseLayer::checkBiasIndex(size_t index) const {
    if (index >= biases.size()) {
      throw exception::InvalidNetworkArchitectureException("Index outside the range of the bias vector");
    }
  }

}; // namespace neuro
