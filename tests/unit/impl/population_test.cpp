#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for Population class") {
  neuro::Population population;

  CHECK(population.size() == 0);
}

TEST_CASE("Check randomization population") {
  std::vector<int> structure = {2, 4, 1};
  neuro::ActivationFunction activation = neuro::maker::makeSigmoid();

  neuro::Population population(1, structure, activation);

  population.randomizeWeights(-5, 5);
  population.randomizeBiases(-5, 5);
}
