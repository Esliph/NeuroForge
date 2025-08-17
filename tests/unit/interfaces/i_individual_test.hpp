#include <doctest/doctest.h>

#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_individual.hpp"

#define TEST_IMPL_IINDIVIDUAL(NAME, TYPE)                                 \
  TEST_CASE(NAME " - Testing implementation for IIndividual interface") { \
    runTestInterfaceIIndividual<TYPE>();                                  \
  }

template <typename IIndividualImpl>
void runTestInterfaceIIndividual() {
  SUBCASE("Fitness") {
    IIndividualImpl individual;

    CHECK(individual.getFitness() == doctest::Approx(0.0f));

    individual.setFitness(10.0f);

    CHECK(individual.getFitness() == doctest::Approx(10.0f));
  }
}
