#pragma once

#include <cmath>
#include <functional>

namespace neuro {

using ActivationHandler = std::function<float(float)>;

struct ActivationFunction {
  ActivationHandler activate;
  ActivationHandler derivate;
};

extern ActivationFunction sigmoid;
extern ActivationFunction relu;
extern ActivationFunction tanh_fn;
extern ActivationFunction leaky_relu;
extern ActivationFunction elu;
extern ActivationFunction swish;
extern ActivationFunction softplus;
extern ActivationFunction hard_sigmoid;

};  // namespace neuro
