#include <iostream>

#include "neuro/activation.hpp"
#include "neuro/neural_network.hpp"
#include "neuro/population.hpp"

int main() {
  neuro::Population population(100, {3, 4, 2}, neuro::makeSigmoid());

  return 0;
}
