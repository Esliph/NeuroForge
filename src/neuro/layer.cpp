#include "neuro/layer.hpp"

#include <random>
#include <vector>

#include "neuro/activation.hpp"

namespace neuro {

Layer::Layer(const layer_weight_t& weights, ActivationFunction activation)
    : weights(weights),
      bias(weights.size()),
      activation(activation) {};

Layer::Layer(const layer_weight_t& weights, const layer_bias_t& bias, ActivationFunction activation)
    : weights(weights),
      bias(bias),
      activation(activation) {};

Layer::Layer(int inputSize, int outputSize, ActivationFunction activation)
    : weights(outputSize, neuro_layer_t(inputSize)),
      bias(outputSize),
      activation(activation) {};

void Layer::loadWeights(float rangeMin, float rangeMax) {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist(rangeMin, rangeMax);

  for (auto& line : weights) {
    for (auto& weight : line) {
      weight = dist(engine);
    }
  }
}

void Layer::loadBias(float rangeMin, float rangeMax) {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::uniform_real_distribution<float> dist(rangeMin, rangeMax);

  for (auto& b : bias) {
    b = dist(engine);
  }
}

neuro_layer_t Layer::process(const neuro_layer_t& inputs) const {
  neuro_layer_t outputs(bias.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    float total = bias[i];

    for (size_t j = 0; j < inputs.size(); ++j) {
      total += weights[i][j] * inputs[j];
    }

    outputs[i] = activation.activate(total);
  }

  return outputs;
}

};  // namespace neuro
