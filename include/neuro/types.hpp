#pragma once

#include <vector>

namespace neuro {

  typedef std::vector<float> neuro_layer_t;

  typedef std::vector<neuro_layer_t> layer_weight_t;
  typedef neuro_layer_t layer_bias_t;

} // namespace neuro
