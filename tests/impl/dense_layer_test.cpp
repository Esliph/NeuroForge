#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

TEST_CASE("Tests for DenseLayer class") {
  neuro::DenseLayer layer(2, 3, neuro::maker::makeSigmoid());

  CHECK(layer.inputSize() == 2);
  CHECK(layer.outputSize() == 3);
}

TEST_CASE("DenseLayer - Feedforward determinístico") {
  neuro::DenseLayer layer;

  layer.setWeights(
      {{0.1f, 0.2f, 0.3f},
       {0.4f, 0.5f, 0.6f}});
  layer.setBiases({0.5f, -0.5f});

  layer.setActivationFunction(neuro::maker::makeRelu());

  neuro::neuro_layer_t input = {1.0f, 2.0f, 3.0f};
  auto output = layer.feedforward(input);

  CHECK(output.size() == 2);
  CHECK(output[0] == doctest::Approx(0.1f * 1 + 0.2f * 2 + 0.3f * 3 + 0.5f));
  CHECK(output[1] == doctest::Approx(std::max(0.0f, 0.4f * 1 + 0.5f * 2 + 0.6f * 3 - 0.5f)));
}

TEST_CASE("DenseLayer - Clone") {
  neuro::DenseLayer original;

  original.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
  original.setBiases({1.0f, -1.0f});

  original.setActivationFunction(neuro::maker::makeSigmoid());

  auto clone = original.clone();

  CHECK(clone->getWeights() == original.getWeights());
  CHECK(clone->getBiases() == original.getBiases());
}

TEST_CASE("DenseLayer - Set/Get peso e bias") {
  neuro::DenseLayer layer(1, 2, neuro::maker::makeSigmoid());

  layer.setWeight(0, 1, 0.75f);
  layer.setBias(1, -0.25f);

  CHECK(layer.weight(0, 1) == doctest::Approx(0.75f));
  CHECK(layer.bias(1) == doctest::Approx(-0.25f));
}

TEST_CASE("DenseLayer - Randomização de pesos e bias") {
  neuro::DenseLayer layer(4, 4);

  layer.randomizeWeights(-1.0f, 1.0f);
  layer.randomizeBiases(-2.0f, 2.0f);

  for (const auto& neurons : layer.getWeights()) {
    for (float weight : neurons) {
      CHECK(weight >= -1.0f);
      CHECK(weight <= 1.0f);
    }
  }

  for (const float bias : layer.getBiases()) {
    CHECK(bias >= -2.0f);
    CHECK(bias <= 2.0f);
  }
}
