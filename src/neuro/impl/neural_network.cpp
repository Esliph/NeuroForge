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

  void NeuralNetwork::addLayer(std::unique_ptr<ILayer> layer) {
    layers.push_back(std::move(layer));
  }

  void NeuralNetwork::reset() {
    for (const auto& layer : layers) {
      layer->reset();
    }
  }

  void NeuralNetwork::clearLayers() {
    layers.clear();
  }

  void NeuralNetwork::removeLayer(size_t index) {
    if (index < layers.size()) {
      layers.erase(layers.begin() + index);
    }
  }

  void NeuralNetwork::popLayer() {
    layers.pop_back();
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

  size_t NeuralNetwork::inputSize() const {
    return layers.empty() ? 0 : layers[0]->inputSize();
  }

  size_t NeuralNetwork::outputSize() const {
    return layers.empty() ? 0 : layers[layers.size() - 1]->outputSize();
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

  void NeuralNetwork::setLayers(std::vector<std::unique_ptr<ILayer>> layers) {
    this->layers = std::move(layers);
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

  const std::vector<std::unique_ptr<ILayer>>& NeuralNetwork::getLayers() const {
    return layers;
  }

  std::vector<std::unique_ptr<ILayer>>& NeuralNetwork::getLayers() {
    return layers;
  }

  const RefProxy<ILayer> NeuralNetwork::layer(size_t index) const {
    if (layers.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return RefProxy<ILayer>(*layers[index]);
  }

  RefProxy<ILayer> NeuralNetwork::layer(size_t index) {
    if (layers.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return RefProxy<ILayer>(*layers[index]);
  }

  size_t NeuralNetwork::sizeLayers() const {
    return layers.size();
  }

  std::vector<std::unique_ptr<ILayer>>::const_iterator NeuralNetwork::begin() const {
    return layers.begin();
  }

  std::vector<std::unique_ptr<ILayer>>::iterator NeuralNetwork::begin() {
    return layers.begin();
  }

  std::vector<std::unique_ptr<ILayer>>::const_iterator NeuralNetwork::end() const {
    return layers.end();
  }

  std::vector<std::unique_ptr<ILayer>>::iterator NeuralNetwork::end() {
    return layers.end();
  }

  neuro_layer_t NeuralNetwork::operator()(const neuro_layer_t& inputs) const {
    return feedforward(inputs);
  }

  const ILayer& NeuralNetwork::operator[](int index) const {
    if (layers.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  ILayer& NeuralNetwork::operator[](int index) {
    if (layers.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Layer vector out-of-range index");
    }

    return *layers[index];
  }

  std::unique_ptr<INeuralNetwork> NeuralNetwork::clone() const {
    return std::make_unique<NeuralNetwork>(*this);
  }

};  // namespace neuro
