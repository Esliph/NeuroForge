#pragma once

#include <memory>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/activation.hpp"
#include "neuro/utils/ref_proxy.hpp"

namespace neuro {

  class DenseLayer : public ILayer {
    layer_weight_t weights{};
    layer_bias_t biases{};

    ActivationFunction activation;

   public:
    DenseLayer() = default;
    DenseLayer(const DenseLayer&) = default;

    DenseLayer(const ActivationFunction& activation);
    DenseLayer(int inputSize, int outputSize);
    DenseLayer(int inputSize, int outputSize, const ActivationFunction& activation);
    DenseLayer(const layer_weight_t& weights, const ActivationFunction& activation);
    DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, const ActivationFunction& activation);

    virtual ~DenseLayer() = default;

    neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

    void reset() override;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    size_t inputSize() const override;
    size_t outputSize() const override;

    RefProxy<float> weight(size_t indexX, size_t indexY) override;
    RefProxy<float> bias(size_t index) override;

    void setActivationFunction(const ActivationFunction&) override;

    void setWeights(const layer_weight_t&) override;
    void setBiases(const layer_bias_t&) override;

    void setWeight(size_t indexX, size_t indexY, float value) override;
    void setBias(size_t index, float value) override;

    const layer_weight_t& getWeights() const override;
    const layer_bias_t& getBiases() const override;

    layer_weight_t& getWeights() override;
    layer_bias_t& getBiases() override;

    const ActivationFunction& getActivationFunction() const override;

    ILayer& operator=(const ILayer&);

    std::unique_ptr<ILayer> clone() const override;
  };

};  // namespace neuro
