#include "neuro/activation.hpp"

namespace neuro {

ActivationFunction sigmoid = {
    [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
    [](float y) { return y * (1.0f - y); }};

ActivationFunction relu = {
    [](float x) { return x > 0 ? x : 0.0f; },
    [](float y) { return y > 0 ? 1.0f : 0.0f; }};

ActivationFunction tanh_fn = {
    [](float x) { return std::tanh(x); },
    [](float y) { return 1.0f - y * y; }};

ActivationFunction leaky_relu = {
    [](float x) { return x > 0 ? x : 0.01f * x; },
    [](float y) { return y > 0 ? 1.0f : 0.01f; }};

ActivationFunction elu = {
    [](float x) { return x >= 0 ? x : std::exp(x) - 1.0f; },
    [](float y) { return y >= 0 ? 1.0f : y + 1.0f; }};

ActivationFunction swish = {
    [](float x) { return x / (1.0f + std::exp(-x)); },
    [](float y) { return y + (1.0f - y) * y; }};

ActivationFunction softplus = {
    [](float x) { return std::log1p(std::exp(x)); },
    [](float y) { return 1.0f - std::exp(-y); }};

ActivationFunction hard_sigmoid = {
    [](float x) { return std::max(0.0f, std::min(1.0f, 0.2f * x + 0.5f)); },
    [](float y) { return (y > 0.0f && y < 1.0f) ? 0.2f : 0.0f; }};

};  // namespace neuro
