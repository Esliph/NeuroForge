#pragma once

#include <functional>
#include <memory>
#include <vector>

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

    void addLayers(std::vector<std::unique_ptr<ILayer>>&& layers) override;
    void addLayer(std::unique_ptr<ILayer> layer) override;
    void addLayer(std::function<std::unique_ptr<ILayer>()> factory, size_t size) override;
    void addLayer(std::function<std::unique_ptr<ILayer>()> factory) override;

    void clear() override;
    void restructure(const std::vector<int>& newStructure) override;

    void clearLayers() override;
    void removeLayer(size_t index) override;

    void popLayer() override;
    void shiftLayer() override;

    size_t inputSize() const override;
    size_t outputSize() const override;

    void setLayers(std::vector<std::unique_ptr<ILayer>> layers) override;
    void setLayer(size_t index, std::unique_ptr<ILayer>) override;

    std::vector<layer_weight_t> getAllWeights() const override;
    std::vector<layer_bias_t> getAllBiases() const override;

    const std::vector<std::unique_ptr<ILayer>>& getLayers() const override;
    std::vector<std::unique_ptr<ILayer>>& getLayers() override;

    const ILayer& layer(size_t index) const override;
    ILayer& layer(size_t index) override;

    size_t sizeLayers() const override;

    bool empty() const override;

    std::vector<std::unique_ptr<ILayer>>::const_iterator begin() const override;
    std::vector<std::unique_ptr<ILayer>>::iterator begin() override;

    std::vector<std::unique_ptr<ILayer>>::const_iterator end() const override;
    std::vector<std::unique_ptr<ILayer>>::iterator end() override;

    neuro_layer_t operator()(const neuro_layer_t& inputs) const override;

    const ILayer& operator[](size_t index) const override;
    ILayer& operator[](size_t index) override;

    std::unique_ptr<INeuralNetwork> clone() const override;
  };

}; // namespace neuro
