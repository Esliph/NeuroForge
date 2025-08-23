#pragma once

namespace neuro {

  class INeuralNetworkParameter {
   public:
    virtual ~INeuralNetworkParameter() = default;

    virtual void randomizeWeights(float min, float max) = 0;
    virtual void randomizeBiases(float min, float max) = 0;
  };

}; // namespace neuro
