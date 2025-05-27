#pragma once

#include <vector>

#include "neuro/activation.hpp"

namespace neuro {

class Layer {
  std::vector<std::vector<float>> weights;
  std::vector<float> bias;

  ActivationFunction activation;

 public:
  Layer(Layer& layer);
  Layer(int inputSize, int outputSize, ActivationFunction activation);
  Layer(std::vector<std::vector<float>>& weights, ActivationFunction activation);
  Layer(std::vector<std::vector<float>>& weights, std::vector<float>& bias, ActivationFunction activation);

  inline void loadWeights(float range = 1.0f) { return loadWeights(-range, range); }
  void loadWeights(float rangeMin, float rangeMax);

  inline void loadBias(float range = 1.0f) { return loadBias(-range, range); }
  void loadBias(float rangeMin, float rangeMax);

  std::vector<float> process(const std::vector<float>& inputs) const;
};

};  // namespace neuro
