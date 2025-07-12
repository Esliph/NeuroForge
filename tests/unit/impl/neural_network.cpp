#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for NeuralNetwork class") {
  neuro::NeuralNetwork network({2, 3, 1}, neuro::maker::makeSigmoid());

  CHECK(network.sizeLayers() == 2);

  CHECK(network[0].inputSize() == 2);
  CHECK(network[0].outputSize() == 3);

  CHECK(network[1].inputSize() == 3);
  CHECK(network[1].outputSize() == 1);
}

TEST_CASE("Tests for NeuralNetwork class 2") {
  neuro::Individual individual;

  CHECK(individual.sizeLayers() == 0);
}
