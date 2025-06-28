#pragma once

#include <cmath>
#include <functional>

namespace neuro {

  using ActivationHandler = std::function<float(float)>;

  struct ActivationFunction {
    ActivationHandler activate;
    ActivationHandler derivate;
  };

};  // namespace neuro
