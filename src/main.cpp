#include <iostream>

#include "neuro/activation.hpp"
#include "neuro/neural_network.hpp"

int main() {
  neuro::NeuralNetwork neural({3, 4, 2}, neuro::makeSigmoid());

  neural.loadWeights(1.f);
  neural.loadBias(1.f);

  auto output = neural.feedforward({1, 2, 3});

  std::cout << output[0] << "\t" << output[1] << "\t" << output[2] << std::endl;

  return 0;
}
