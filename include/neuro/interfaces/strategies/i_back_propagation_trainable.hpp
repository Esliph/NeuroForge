#pragma once

#include <vector>

#include "neuro/interfaces/i_layer.hpp"

namespace neuro {

class IBackPropagationTrainable {
 public:
  virtual ~IBackPropagationTrainable() = default;

  virtual void trainBatch(const neuro_layer_t& inputs, const neuro_layer_t& expectedOutputs, float learningRate, size_t epochs) = 0;
};

};  // namespace neuro
