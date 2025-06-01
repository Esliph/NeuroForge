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

void Layer::mutate(float rate, float strength, std::default_random_engine& engine) {
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);
  std::normal_distribution<float> perturbation(0.0f, strength);

  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      if (chance(engine) < rate) {
        weights[i][j] += perturbation(engine);
      }
    }
  }

  for (float& b : bias) {
    if (chance(engine) < rate) {
      b += perturbation(engine);
    }
  }
}

Layer Layer::crossover(const Layer& partner, std::default_random_engine& engine) const {
  std::uniform_int_distribution<int> choose(0, 1);

  layer_weight_t newWeights = weights;
  layer_bias_t newBias = bias;

  for (size_t i = 0; i < weights.size(); ++i) {
    for (size_t j = 0; j < weights[i].size(); ++j) {
      if (choose(engine) == 1) {
        newWeights[i][j] = partner.weights[i][j];
      }
    }
  }

  for (size_t i = 0; i < weights.size(); ++i) {
    if (choose(engine) == 1) {
      newBias[i] = partner.bias[i];
    }
  }

  return Layer(newWeights, newBias, activation);
}

};  // namespace neuro
