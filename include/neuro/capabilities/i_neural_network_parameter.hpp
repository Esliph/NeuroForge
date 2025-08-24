#pragma once

#include <functional>

namespace neuro {

  class INeuralNetworkParameter {
   public:
    virtual ~INeuralNetworkParameter() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;

    virtual void mutateWeights(const std::function<float(float)>& mutator) = 0;
    virtual void mutateBiases(const std::function<float(float)>& mutator) = 0;
  };

}; // namespace neuro
