#include <doctest/doctest.h>

#include "interfaces/i_individual_test.hpp"
#include "neuro/neuro.hpp"

TEST_CASE("Individual - Object construction tests") {
  SUBCASE("Create Individual without parameters") {
    neuro::Individual individual;

    CHECK(individual.getNeuralNetwork().sizeLayers() == 0);
    CHECK(individual.getNeuralNetwork().inputSize() == 0);
    CHECK(individual.getNeuralNetwork().outputSize() == 0);

    CHECK(individual.getFitness() == doctest::Approx(0.0f));
  }
}

TEST_IMPL_IINDIVIDUAL("Individual", neuro::Individual);
