#include "neuro/impl/neural_network.hpp"

#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/types.hpp"
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

  void NeuralNetwork::clear() {
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->clear();
    }
  }

  void NeuralNetwork::restructure(const std::vector<int>& newStructure) {
    clearLayers();

    for (size_t i = 0; i < newStructure.size() - 1; i++) {
      layers.emplace_back(std::make_unique<DenseLayer>(newStructure[i], newStructure[i + 1]));
    }
  }

  FORCE_INLINE void NeuralNetwork::clearLayers() {
    layers.clear();
  }

  void NeuralNetwork::removeLayer(size_t index) {
    if (index < layers.size()) {
      layers.erase(layers.begin() + index);
    }
  }

  FORCE_INLINE void NeuralNetwork::popLayer() {
    layers.pop_back();
  }

  FORCE_INLINE void NeuralNetwork::shiftLayer() {
    layers.erase(layers.begin());
  }

  FORCE_INLINE size_t NeuralNetwork::inputSize() const {
    return layers.empty() ? 0 : layers[0]->inputSize();
  }

  FORCE_INLINE size_t NeuralNetwork::outputSize() const {
    return layers.empty() ? 0 : layers[layers.size() - 1]->outputSize();
  }

  FORCE_INLINE void NeuralNetwork::setLayers(std::vector<std::unique_ptr<ILayer>> layers) {
    this->layers = std::move(layers);
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

  void NeuralNetwork::mutateWeights(const std::function<float(float)>& mutator) {
    for (auto& layer : layers) {
      layer->mutateWeights(mutator);
    }
  }

  void NeuralNetwork::mutateBiases(const std::function<float(float)>& mutator) {
    for (auto& layer : layers) {
      layer->mutateBiases(mutator);
    }
  }

  FORCE_INLINE void NeuralNetwork::addLayers(std::vector<std::unique_ptr<ILayer>>&& layers) {
    for (size_t i = 0; i < layers.size(); i++) {
      this->layers.push_back(std::move(layers[i]));
    }
  }

  FORCE_INLINE void NeuralNetwork::addLayer(std::unique_ptr<ILayer> layer) {
    layers.push_back(std::move(layer));
  }

  FORCE_INLINE void NeuralNetwork::addLayer(std::function<std::unique_ptr<ILayer>()> factory, size_t size) {
    for (size_t i = 0; i < size; i++) {
      layers.push_back(factory());
    }
  }

  FORCE_INLINE void NeuralNetwork::addLayer(std::function<std::unique_ptr<ILayer>()> factory) {
    layers.push_back(factory());
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

  FORCE_INLINE const std::vector<std::unique_ptr<ILayer>>& NeuralNetwork::getLayers() const {
    return layers;
  }

  FORCE_INLINE std::vector<std::unique_ptr<ILayer>>& NeuralNetwork::getLayers() {
    return layers;
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

  FORCE_INLINE size_t NeuralNetwork::sizeLayers() const {
    return layers.size();
  }

  FORCE_INLINE bool NeuralNetwork::empty() const {
    return layers.empty();
  }

  FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator NeuralNetwork::begin() const {
    return layers.begin();
  }

  FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator NeuralNetwork::begin() {
    return layers.begin();
  }

  FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator NeuralNetwork::end() const {
    return layers.end();
  }

  FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator NeuralNetwork::end() {
    return layers.end();
  }

  FORCE_INLINE neuro_layer_t NeuralNetwork::operator()(const neuro_layer_t& inputs) const {
    return feedforward(inputs);
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

  FORCE_INLINE std::unique_ptr<INeuralNetwork> NeuralNetwork::clone() const {
    return std::make_unique<NeuralNetwork>(*this);
  }

}; // namespace neuro
