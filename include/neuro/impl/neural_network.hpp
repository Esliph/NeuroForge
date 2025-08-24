#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class NeuralNetwork : public INeuralNetwork {
    std::vector<std::unique_ptr<ILayer>> layers{};

   public:
    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&);
    NeuralNetwork(NeuralNetwork&&) noexcept = default;

    NeuralNetwork(std::initializer_list<ILayer*> layers);
    NeuralNetwork(const std::vector<ILayer*>& layers);

    NeuralNetwork(std::vector<std::unique_ptr<ILayer>>& layers);
    NeuralNetwork(std::vector<std::unique_ptr<ILayer>>&& layers);

    NeuralNetwork(const std::vector<std::function<std::unique_ptr<ILayer>()>>& factories);
    NeuralNetwork(const std::function<std::unique_ptr<ILayer>()>& factory, size_t size = 1);

    NeuralNetwork(const std::vector<int>& structure);
    NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation);
    NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    NeuralNetwork(const ILayer& prototype, size_t size);

    virtual ~NeuralNetwork() = default;

    neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    void mutateWeights(const std::function<float(float)>& mutator) override;
    void mutateBiases(const std::function<float(float)>& mutator) override;

    FORCE_INLINE void addLayers(std::vector<std::unique_ptr<ILayer>>&& layers) {
      for (size_t i = 0; i < layers.size(); i++) {
        this->layers.push_back(std::move(layers[i]));
      }
    }

    FORCE_INLINE void addLayer(std::unique_ptr<ILayer> layer) {
      layers.push_back(std::move(layer));
    }

    FORCE_INLINE void addLayer(std::function<std::unique_ptr<ILayer>()> factory, size_t size) {
      for (size_t i = 0; i < size; i++) {
        layers.push_back(factory());
      }
    }

    FORCE_INLINE void addLayer(std::function<std::unique_ptr<ILayer>()> factory) {
      layers.push_back(factory());
    }

    void clear() override;
    void restructure(const std::vector<int>& newStructure) override;

    FORCE_INLINE void clearLayers() {
      layers.clear();
    }

    void removeLayer(size_t index) override;

    FORCE_INLINE void popLayer() {
      layers.pop_back();
    }

    FORCE_INLINE void shiftLayer() {
      layers.erase(layers.begin());
    }

    FORCE_INLINE size_t inputSize() const {
      return layers.empty() ? 0 : layers[0]->inputSize();
    }

    FORCE_INLINE size_t outputSize() const {
      return layers.empty() ? 0 : layers[layers.size() - 1]->outputSize();
    }

    FORCE_INLINE void setLayers(std::vector<std::unique_ptr<ILayer>> layers) {
      this->layers = std::move(layers);
    }

    void setLayer(size_t index, std::unique_ptr<ILayer>) override;

    std::vector<layer_weight_t> getAllWeights() const override;
    std::vector<layer_bias_t> getAllBiases() const override;

    FORCE_INLINE const std::vector<std::unique_ptr<ILayer>>& getLayers() const {
      return layers;
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>& getLayers() {
      return layers;
    }

    const ILayer& layer(size_t index) const override;
    ILayer& layer(size_t index) override;

    FORCE_INLINE size_t sizeLayers() const {
      return layers.size();
    }

    FORCE_INLINE bool empty() const {
      return layers.empty();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const {
      return layers.begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator begin() {
      return layers.begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator end() const {
      return layers.end();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator end() {
      return layers.end();
    }

    FORCE_INLINE neuro_layer_t operator()(const neuro_layer_t& inputs) const {
      return feedforward(inputs);
    }

    const ILayer& operator[](size_t index) const override;
    ILayer& operator[](size_t index) override;

    FORCE_INLINE std::unique_ptr<INeuralNetwork> clone() const {
      return std::make_unique<NeuralNetwork>(*this);
    }
  };

}; // namespace neuro
