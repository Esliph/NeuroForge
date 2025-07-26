#pragma once

#include <memory>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class DenseLayer : public ILayer {
    layer_weight_t weights{};
    layer_bias_t biases{};

    ActivationFunction activation;

   public:
    DenseLayer() = default;
    DenseLayer(const DenseLayer&) = default;

    DenseLayer(const ActivationFunction& activation);
    DenseLayer(size_t inputSize, size_t outputSize);
    DenseLayer(size_t inputSize, size_t outputSize, const ActivationFunction& activation);
    DenseLayer(const layer_weight_t& weights, const ActivationFunction& activation);
    DenseLayer(const layer_weight_t& weights, const layer_bias_t& biases, const ActivationFunction& activation);

    virtual ~DenseLayer() = default;

    neuro_layer_t feedforward(const neuro_layer_t& inputs) const override;

    void reset() override;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    FORCE_INLINE size_t inputSize() const override {
      return weights.empty() ? 0 : weights[0].size();
    }

    FORCE_INLINE size_t outputSize() const override {
      return weights.size();
    }

    float& weight(size_t indexX, size_t indexY) override;
    float& bias(size_t index) override;

    const float& weight(size_t indexX, size_t indexY) const override;
    const float& bias(size_t index) const override;

    FORCE_INLINE void setActivationFunction(const ActivationFunction& activation) override {
      this->activation = activation;
    }

    FORCE_INLINE void setWeights(const layer_weight_t& weights) override {
      this->weights = weights;
    }

    FORCE_INLINE void setBiases(const layer_bias_t& biases) override {
      this->biases = biases;
    }

    void setWeight(size_t indexX, size_t indexY, float value) override;
    void setBias(size_t index, float value) override;

    FORCE_INLINE const layer_weight_t& getWeights() const override {
      return weights;
    }

    FORCE_INLINE const layer_bias_t& getBiases() const override {
      return biases;
    }

    FORCE_INLINE layer_weight_t& getWeights() override {
      return weights;
    }

    FORCE_INLINE layer_bias_t& getBiases() override {
      return biases;
    }

    FORCE_INLINE const ActivationFunction& getActivationFunction() const override {
      return activation;
    }

    ILayer& operator=(const ILayer&);

    FORCE_INLINE std::unique_ptr<ILayer> clone() const override {
      return std::make_unique<DenseLayer>(*this);
    }

   private:
    void checkWeightIndex(size_t indexX, size_t indexY) const;
    void checkBiasIndex(size_t index) const;
  };

};  // namespace neuro
