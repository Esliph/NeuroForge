#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for DenseLayer class") {
  neuro::DenseLayer layer(2, 3, neuro::makeSigmoid());

  CHECK(layer.inputSize() == 2);
  CHECK(layer.outputSize() == 3);
}
