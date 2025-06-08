#pragma once

#include <vector>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/strategies/i_strategy_evolution.hpp"

namespace neuro {

class BackPropagationTrainable : public IStrategyEvolution {
 public:
  BackPropagationTrainable() = default;
  virtual ~BackPropagationTrainable() = default;

  virtual void trainBatch(const neuro_layer_t& inputs, const neuro_layer_t& expectedOutputs, float learningRate, size_t epochs) = 0;
};

};  // namespace neuro
