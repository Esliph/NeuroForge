#pragma once

#include <random>
#include <vector>

#include "neuro/activation.hpp"

namespace neuro {

typedef std::vector<float> neuro_layer_t;

typedef std::vector<neuro_layer_t> layer_weight_t;
typedef neuro_layer_t layer_bias_t;

class Layer {
  layer_weight_t weights{};
  layer_bias_t bias{};

  ActivationFunction activation;

 public:
  Layer() = default;
  Layer(const Layer&) = default;
  Layer(int inputSize, int outputSize, ActivationFunction activation);
  Layer(const layer_weight_t& weights, ActivationFunction activation);
  Layer(const layer_weight_t& weights, const layer_bias_t& bias, ActivationFunction activation);

  inline void loadWeights(float range = 1.0f) { return loadWeights(-range, range); }
  void loadWeights(float rangeMin, float rangeMax);

  inline void loadBias(float range = 1.0f) { return loadBias(-range, range); }
  void loadBias(float rangeMin, float rangeMax);

  neuro_layer_t process(const neuro_layer_t& inputs) const;

  void mutate(float rate, float strength, std::default_random_engine& engine);

  Layer& operator=(const Layer&) = default;
};

};  // namespace neuro
