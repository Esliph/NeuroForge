#include "neuro/impl/neural_network.hpp"

#include <memory>
#include <vector>

#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/interfaces/layer/i_layer.hpp"
#include "neuro/utils/activation.hpp"
#include "neuro/utils/interfaces/i_iterable.hpp"

namespace neuro {

NeuralNetwork::NeuralNetwork(const std::vector<std::unique_ptr<ILayer>>& layers)
    : INeuralNetwork() {
  for (auto& layer : layers) {
    this->layers.push_back(std::move(layer));
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

void NeuralNetwork::addLayer(std::unique_ptr<ILayer> layer) {
  layers.push_back(std::move(layer));
}

void NeuralNetwork::clearLayers() {
  layers.clear();
}

std::vector<std::unique_ptr<ILayer>> NeuralNetwork::clearLayersAndReturn() {
  return std::move(layers);
}

void NeuralNetwork::randomizeAll(float minWeight, float maxWeight, float minBias, float maxBias) {
  for (auto& layer : layers) {
    layer->randomizeWeights(minWeight, maxWeight);
    layer->randomizeBiases(minBias, maxBias);
  }
}

void NeuralNetwork::randomizeWeights(float min, float max) {
  for (auto& layer : layers) {
    layer->randomizeWeights(min, max);
  }
}

void NeuralNetwork::randomizeBiases(float min, float max) {
  for (auto& layer : layers) {
    layer->randomizeBiases(min, max);
  }
}

void NeuralNetwork::setAllWeights(const std::vector<layer_weight_t>& weights) {
  for (size_t i = 0; i < layers.size() && i < weights.size(); i++) {
    layers[i]->setWeights(weights[i]);
  }
}

void NeuralNetwork::setAllBiases(const std::vector<layer_bias_t>& biases) {
  for (size_t i = 0; i < layers.size() && i < biases.size(); i++) {
    layers[i]->setBiases(biases[i]);
  }
}

const ILayer& NeuralNetwork::getLayer(size_t index) const {
  return *layers[index];
}

size_t NeuralNetwork::getNumLayers() const {
  return layers.size();
}

};  // namespace neuro
