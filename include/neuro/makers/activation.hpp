#pragma once

#include <cmath>
#include <functional>

#include "internal/attribute.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  namespace maker {

    FORCE_INLINE ActivationFunction makeSigmoid() {
      return {
          [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
          [](float y) { return y * (1.0f - y); }};
    }

    FORCE_INLINE ActivationFunction makeRelu() {
      return {
          [](float x) { return x > 0 ? x : 0.0f; },
          [](float y) { return y > 0 ? 1.0f : 0.0f; }};
    }

    FORCE_INLINE ActivationFunction makeTanh_fn() {
      return {
          [](float x) { return std::tanh(x); },
          [](float y) { return 1.0f - y * y; }};
    }

    FORCE_INLINE ActivationFunction makeLeaky_relu() {
      return {
          [](float x) { return x > 0 ? x : 0.01f * x; },
          [](float y) { return y > 0 ? 1.0f : 0.01f; }};
    }

    FORCE_INLINE ActivationFunction makeElu() {
      return {
          [](float x) { return x >= 0 ? x : std::exp(x) - 1.0f; },
          [](float y) { return y >= 0 ? 1.0f : y + 1.0f; }};
    }

    FORCE_INLINE ActivationFunction makeSwish() {
      return {
          [](float x) { return x / (1.0f + std::exp(-x)); },
          [](float y) { return y + (1.0f - y) * y; }};
    }

    FORCE_INLINE ActivationFunction makeSoftplus() {
      return {
          [](float x) { return std::log1p(std::exp(x)); },
          [](float y) { return 1.0f - std::exp(-y); }};
    }

    FORCE_INLINE ActivationFunction makeHard_sigmoid() {
      return {
          [](float x) { return std::max(0.0f, std::min(1.0f, 0.2f * x + 0.5f)); },
          [](float y) { return (y > 0.0f && y < 1.0f) ? 0.2f : 0.0f; }};
    }

    FORCE_INLINE ActivationFunction makeIdentity() {
      return {
          [](float x) { return x; },
          [](float) { return 1.0f; }};
    }

  };  // namespace maker

};  // namespace neuro
