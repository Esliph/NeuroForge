#pragma once

#include <functional>

#include "neuro/types.hpp"

namespace neuro {

  class ILayerWeight {
   public:
    virtual ~ILayerWeight() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual void mutateWeights(const std::function<float(float)>& mutator) = 0;
    virtual void mutateBiases(const std::function<float(float)>& mutator) = 0;

    virtual float meanWeight() const = 0;
    virtual float meanBias() const = 0;

    virtual float& weightRef(size_t indexX, size_t indexY) = 0;
    virtual float& biasRef(size_t index) = 0;

    virtual const float& weightRef(size_t indexX, size_t indexY) const = 0;
    virtual const float& biasRef(size_t index) const = 0;

    virtual float getWeight(size_t indexX, size_t indexY) const = 0;
    virtual float getBias(size_t index) const = 0;

    virtual void setWeight(size_t indexX, size_t indexY, float value) = 0;
    virtual void setBias(size_t index, float value) = 0;

    virtual layer_weight_t& getWeights() = 0;
    virtual layer_bias_t& getBiases() = 0;

    virtual const layer_weight_t& getWeights() const = 0;
    virtual const layer_bias_t& getBiases() const = 0;

    virtual void setWeights(const layer_weight_t&) = 0;
    virtual void setBiases(const layer_bias_t&) = 0;
  };

}; // namespace neuro
