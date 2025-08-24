#pragma once

#include <functional>
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

    FORCE_INLINE void clear() {
      reshape(inputSize(), outputSize());
    }

    void reshape(size_t newInputSize, size_t newOutputSize) override;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    void mutateWeights(const std::function<float(float)>& mutator) override;
    void mutateBiases(const std::function<float(float)>& mutator) override;

    FORCE_INLINE bool validateInternalShape() {
      return validateInternalShape(weights, biases);
    }

    float meanWeight() const override;
    float meanBias() const override;

    FORCE_INLINE size_t inputSize() const {
      if (weights.empty())
        UNLIKELY {
          return 0;
        }

      return weights[0].size();
    }

    FORCE_INLINE size_t outputSize() const {
      return weights.size();
    }

    float& weightRef(size_t indexX, size_t indexY) override;
    float& biasRef(size_t index) override;

    const float& weightRef(size_t indexX, size_t indexY) const override;
    const float& biasRef(size_t index) const override;

    FORCE_INLINE const ActivationFunction& getActivationFunction() const {
      return activation;
    }

    FORCE_INLINE void setActivationFunction(const ActivationFunction& activation) {
      this->activation = activation;
    }

    float getWeight(size_t indexX, size_t indexY) const override;
    float getBias(size_t index) const override;

    void setWeight(size_t indexX, size_t indexY, float value) override;
    void setBias(size_t index, float value) override;

    FORCE_INLINE layer_weight_t& getWeights() {
      return weights;
    }

    FORCE_INLINE layer_bias_t& getBiases() {
      return biases;
    }

    FORCE_INLINE const layer_weight_t& getWeights() const {
      return weights;
    }

    FORCE_INLINE const layer_bias_t& getBiases() const {
      return biases;
    }

    FORCE_INLINE void setWeights(const layer_weight_t& weights) {
      this->weights = weights;
    }

    FORCE_INLINE void setBiases(const layer_bias_t& biases) {
      this->biases = biases;
    }

    FORCE_INLINE std::unique_ptr<ILayer> clone() const {
      return std::make_unique<DenseLayer>(*this);
    }

    ILayer& operator=(const ILayer&);

   private:
    virtual bool validateInternalShape(const layer_weight_t& weights, const layer_bias_t& biases);

    virtual void checkWeightIndex(size_t indexX, size_t indexY) const;
    virtual void checkBiasIndex(size_t index) const;
  };

}; // namespace neuro
