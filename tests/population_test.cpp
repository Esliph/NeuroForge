#include <doctest/doctest.h>

#include <iostream>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for Population class") {
  neuro::Population population;

  CHECK(population.size() == 0);
}
