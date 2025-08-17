#include "neuro/impl/neural_network.hpp"

#include <memory>
#include <vector>

#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  NeuralNetwork::NeuralNetwork(const NeuralNetwork& neuralNetwork)
    : INeuralNetwork() {
    for (size_t i = 0; i < neuralNetwork.sizeLayers(); i++) {
      layers.push_back(neuralNetwork[i].clone());
    }
  }

  NeuralNetwork::NeuralNetwork(std::initializer_list<ILayer*> layers)
    : INeuralNetwork() {
    for (const auto* layer : layers) {
      this->layers.push_back(layer->clone());
    }
  }

  NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<ILayer>>& layers)
    : INeuralNetwork() {
    for (size_t i = 0; i < layers.size(); i++) {
      this->layers.push_back(std::move(layers[i]));
    }
  }

  NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<ILayer>>&& layers)
    : INeuralNetwork(),
      layers(std::move(layers)) {}

  NeuralNetwork::NeuralNetwork(const std::vector<ILayer*>& layers)
    : INeuralNetwork() {
    for (size_t i = 0; i < layers.size(); i++) {
      this->layers.push_back(layers[i]->clone());
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<std::function<std::unique_ptr<ILayer>()>>& factories) {
    for (auto& f : factories) {
      layers.push_back(f());
    }
  }

  NeuralNetwork::NeuralNetwork(const std::function<std::unique_ptr<ILayer>()>& factory, size_t size) {
    for (size_t i = 0; i < size; i++) {
      layers.push_back(factory());
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<int>& structure)
    : INeuralNetwork() {
    for (size_t i = 0; i < structure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(structure[i], structure[i + 1]));
    }
  }

  NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation)
    : INeuralNetwork() {
    for (size_t i = 0; i < structure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(structure[i], structure[i + 1], activation));
    }
  }

  NeuralNetwork::NeuralNetwork(const ILayer& prototype, size_t size) {
    for (size_t i = 0; i < size; i++) {
      layers.emplace_back(prototype.clone());
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

    for (size_t i = 0; i < layers.size(); i++) {
      current = layers[i]->feedforward(current);
    }

    return current;
  }

  void NeuralNetwork::reset() {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->reset();
    }
  }

  void NeuralNetwork::reset(const std::vector<int>& newStructure) {
    clearLayers();

    for (size_t i = 0; i < newStructure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(newStructure[i], newStructure[i + 1]));
    }
  }

  void NeuralNetwork::removeLayer(size_t index) {
    if (index < layers.size()) {
      layers.erase(layers.begin() + index);
    }
  }

  void NeuralNetwork::randomizeWeights(float min, float max) {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->randomizeWeights(min, max);
    }
  }

  void NeuralNetwork::randomizeBiases(float min, float max) {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->randomizeBiases(min, max);
    }
  }

  void NeuralNetwork::setLayer(size_t index, std::unique_ptr<ILayer> layer) {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    layers[index] = std::move(layer);
  }

  std::vector<layer_weight_t> NeuralNetwork::getAllWeights() const {
    std::vector<layer_weight_t> allWeights;

    for (size_t i = 0; i < layers.size(); i++) {
      allWeights.push_back(layers[i]->getWeights());
    }

    return allWeights;
  }

  std::vector<layer_bias_t> NeuralNetwork::getAllBiases() const {
    std::vector<layer_bias_t> allBiases;

    for (size_t i = 0; i < layers.size(); i++) {
      allBiases.push_back(layers[i]->getBiases());
    }

    return allBiases;
  }

  const ILayer& NeuralNetwork::layer(size_t index) const {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  ILayer& NeuralNetwork::layer(size_t index) {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  const ILayer& NeuralNetwork::operator[](size_t index) const {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  ILayer& NeuralNetwork::operator[](size_t index) {
    if (index >= layers.size()) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

}; // namespace neuro
