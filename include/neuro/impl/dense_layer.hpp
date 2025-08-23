#pragma once

#include <memory>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/makers/activation.hpp"
#include "neuro/types.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class DenseLayer : public ILayer {
    layer_weight_t weights{};
    layer_bias_t biases{};

    ActivationFunction activation = neuro::maker::activationIdentity();

   public:
    DenseLayer() = default;
    DenseLayer(const DenseLayer&) = default;

    DenseLayer(const layer_weight_t& weights);
    DenseLayer(const layer_bias_t& biases);
    DenseLayer(const ActivationFunction& activation);

    DenseLayer(size_t inputSize, size_t outputSize);
    DenseLayer(size_t inputSize, size_t outputSize, const ActivationFunction& activation);

    DenseLayer(const layer_weight_t& weights, const ActivationFunction& activation);
    DenseLayer(const layer_bias_t& biases, const ActivationFunction& activation);

    DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases);
    DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, const ActivationFunction& activation);

    virtual ~DenseLayer() = default;

    neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

    void clear() override;
    void reshape(size_t newInputSize, size_t newOutputSize) override;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    bool validateInternalShape() override;

    float meanWeight() const override;
    float meanBias() const override;

    size_t inputSize() const override;
    size_t outputSize() const override;

    float& weightRef(size_t indexX, size_t indexY) override;
    float& biasRef(size_t index) override;

    const float& weightRef(size_t indexX, size_t indexY) const override;
    const float& biasRef(size_t index) const override;

    const ActivationFunction& getActivationFunction() const override;
    void setActivationFunction(const ActivationFunction& activation) override;

    float getWeight(size_t indexX, size_t indexY) const override;
    float getBias(size_t index) const override;

    void setWeight(size_t indexX, size_t indexY, float value) override;
    void setBias(size_t index, float value) override;

    layer_weight_t& getWeights() override;
    layer_bias_t& getBiases() override;

    const layer_weight_t& getWeights() const override;
    const layer_bias_t& getBiases() const override;

    void setWeights(const layer_weight_t& weights) override;
    void setBiases(const layer_bias_t& biases) override;

    std::unique_ptr<ILayer> clone() const override;

    ILayer& operator=(const ILayer&);

   private:
    virtual bool validateInternalShape(const layer_weight_t& weights, const layer_bias_t& biases);

    virtual void checkWeightIndex(size_t indexX, size_t indexY) const;
    virtual void checkBiasIndex(size_t index) const;
  };

}; // namespace neuro
