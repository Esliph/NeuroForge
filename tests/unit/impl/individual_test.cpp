#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for Individual class") {
  neuro::Individual individual({2, 3, 1}, neuro::maker::makeSigmoid());

  individual.randomizeBiases(-1.0f, 1.0f);

  CHECK(individual.sizeLayers() == 2);

  CHECK(individual[0].inputSize() == 2);
  CHECK(individual[0].outputSize() == 3);

  CHECK(individual[1].inputSize() == 3);
  CHECK(individual[1].outputSize() == 1);

  CHECK(individual.getFitness() == 0.0f);

  individual.evaluateFitness([]([[maybe_unused]] const neuro::INeuralNetwork& network) {
    return 10.0f;
  });

  CHECK(individual.getFitness() == 10.0f);
}
