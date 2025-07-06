#pragma once

#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class NeuralNetwork : public INeuralNetwork {
    std::vector<std::unique_ptr<ILayer>> layers{};

   public:
    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&);

    NeuralNetwork(std::vector<std::unique_ptr<ILayer>>& layers);
    NeuralNetwork(const std::vector<ILayer>& layers);
    NeuralNetwork(const std::vector<int>& structure, const ActivationFunction& activation);
    NeuralNetwork(const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    virtual ~NeuralNetwork() = default;

    neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

    FORCE_INLINE void addLayer(std::unique_ptr<ILayer> layer) override {
      layers.push_back(std::move(layer));
    }

    FORCE_INLINE void addLayer(ILayer* layer) override {
      layers.push_back(std::unique_ptr<ILayer>(layer));
    }

    void reset() override;

    FORCE_INLINE void clearLayers() override {
      layers.clear();
    }

    FORCE_INLINE void removeLayer(size_t index) override;

    FORCE_INLINE void popLayer() override {
      layers.pop_back();
    }

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    FORCE_INLINE size_t inputSize() const override {
      return layers.empty() ? 0 : layers[0]->inputSize();
    }

    FORCE_INLINE size_t outputSize() const override {
      return layers.empty() ? 0 : layers[layers.size() - 1]->outputSize();
    }

    void setAllWeights(const std::vector<layer_weight_t>&) override;
    void setAllBiases(const std::vector<layer_bias_t>&) override;

    FORCE_INLINE void setLayers(std::vector<std::unique_ptr<ILayer>> layers) override {
      this->layers = std::move(layers);
    }

    std::vector<layer_weight_t> getAllWeights() const override;
    std::vector<layer_bias_t> getAllBiases() const override;

    FORCE_INLINE const std::vector<std::unique_ptr<ILayer>>& getLayers() const override {
      return layers;
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>& getLayers() override {
      return layers;
    }

    const RefProxy<ILayer> layer(size_t index) const override;
    RefProxy<ILayer> layer(size_t index) override;

    FORCE_INLINE size_t sizeLayers() const override {
      return layers.size();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const override {
      return layers.begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator begin() override {
      return layers.begin();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::const_iterator end() const override {
      return layers.end();
    }

    FORCE_INLINE std::vector<std::unique_ptr<ILayer>>::iterator end() override {
      return layers.end();
    }

    FORCE_INLINE neuro_layer_t operator()(const neuro_layer_t& inputs) const override {
      return feedforward(inputs);
    }

    const ILayer& operator[](size_t index) const override;
    ILayer& operator[](size_t index) override;

    FORCE_INLINE std::unique_ptr<INeuralNetwork> clone() const override {
      return std::make_unique<NeuralNetwork>(*this);
    }
  };

};  // namespace neuro
