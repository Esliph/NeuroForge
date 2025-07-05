#include "neuro/impl/neural_network.hpp"

#include <memory>
#include <vector>

#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/activation.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
      : INeuralNetwork() {
    for (const auto& layer : neuralNetwork.layers) {
      layers.push_back(std::move(layer->clone()));
    }
  }

  NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<ILayer>>& layers)
      : INeuralNetwork() {
    for (auto& layer : layers) {
      this->layers.push_back(std::move(layer));
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<ILayer>& layers)
      : INeuralNetwork() {
    for (auto& layer : layers) {
      this->layers.push_back(std::move(layer.clone()));
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation) : INeuralNetwork() {
    for (size_t i = 0; i < structure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(structure[i], structure[i + 1], activation));
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations) {
    for (size_t i = 0; i < structure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(structure[i], structure[i + 1], activations[i]));
    }
  }

  neuro_layer_t NeuralNetwork::feedforward(const neuro_layer_t& inputs) const {
    if (inputSize() != inputs.size()) {
      throw exception::InvalidNetworkArchitectureException("Amount of data input does not match neuron data input");
    }

    neuro_layer_t current = inputs;

    for (const auto& layer : layers) {
      current = layer->feedforward(current);
    }

    return current;
  }

  void NeuralNetwork::reset() {
    for (const auto& layer : layers) {
      layer->reset();
    }
  }

  void NeuralNetwork::removeLayer(size_t index) {
    if (index < layers.size()) {
      layers.erase(layers.begin() + index);
    }
  }

  void NeuralNetwork::randomizeWeights(float min, float max) {
    for (const auto& layer : layers) {
      layer->randomizeWeights(min, max);
    }
  }

  void NeuralNetwork::randomizeBiases(float min, float max) {
    for (const auto& layer : layers) {
      layer->randomizeBiases(min, max);
    }
  }

  void NeuralNetwork::setAllWeights(const std::vector<layer_weight_t>& allWeights) {
    for (size_t i = 0; i < layers.size() && i < allWeights.size(); i++) {
      layers[i]->setWeights(allWeights[i]);
    }
  }

  void NeuralNetwork::setAllBiases(const std::vector<layer_bias_t>& allBiases) {
    for (size_t i = 0; i < layers.size() && i < allBiases.size(); i++) {
      layers[i]->setBiases(allBiases[i]);
    }
  }

  std::vector<layer_weight_t> NeuralNetwork::getAllWeights() const {
    std::vector<layer_weight_t> allWeights;

    for (const auto& layer : layers) {
      allWeights.push_back(layer->getWeights());
    }

    return allWeights;
  }

  std::vector<layer_bias_t> NeuralNetwork::getAllBiases() const {
    std::vector<layer_bias_t> allBiases;

    for (const auto& layer : layers) {
      allBiases.push_back(layer->getBiases());
    }

    return allBiases;
  }

  const RefProxy<ILayer> NeuralNetwork::layer(size_t index) const {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return RefProxy<ILayer>(*layers[index]);
  }

  RefProxy<ILayer> NeuralNetwork::layer(size_t index) {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return RefProxy<ILayer>(*layers[index]);
  }

  const ILayer& NeuralNetwork::operator[](int index) const {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  ILayer& NeuralNetwork::operator[](int index) {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

};  // namespace neuro
