#pragma once

#include <cmath>
#include <functional>

namespace neuro {

  using ActivationHandler = std::function<float(float)>;

  struct ActivationFunction {
    ActivationHandler activate;
    ActivationHandler derivate;
  };

  inline ActivationFunction makeSigmoid() {
    return {
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
        [](float y) { return y * (1.0f - y); }};
  }

  inline ActivationFunction makeRelu() {
    return {
        [](float x) { return x > 0 ? x : 0.0f; },
        [](float y) { return y > 0 ? 1.0f : 0.0f; }};
  }

  inline ActivationFunction makeTanh_fn() {
    return {
        [](float x) { return std::tanh(x); },
        [](float y) { return 1.0f - y * y; }};
  }

  inline ActivationFunction makeLeaky_relu() {
    return {
        [](float x) { return x > 0 ? x : 0.01f * x; },
        [](float y) { return y > 0 ? 1.0f : 0.01f; }};
  }

  inline ActivationFunction makeElu() {
    return {
        [](float x) { return x >= 0 ? x : std::exp(x) - 1.0f; },
        [](float y) { return y >= 0 ? 1.0f : y + 1.0f; }};
  }

  inline ActivationFunction makeSwish() {
    return {
        [](float x) { return x / (1.0f + std::exp(-x)); },
        [](float y) { return y + (1.0f - y) * y; }};
  }

  inline ActivationFunction makeSoftplus() {
    return {
        [](float x) { return std::log1p(std::exp(x)); },
        [](float y) { return 1.0f - std::exp(-y); }};
  }

  inline ActivationFunction makeHard_sigmoid() {
    return {
        [](float x) { return std::max(0.0f, std::min(1.0f, 0.2f * x + 0.5f)); },
        [](float y) { return (y > 0.0f && y < 1.0f) ? 0.2f : 0.0f; }};
  }

};  // namespace neuro
