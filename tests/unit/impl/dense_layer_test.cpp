#include <doctest/doctest.h>

#include "neuro/neuro.hpp"

void checkEqualLayers(neuro::ILayer& layerA, neuro::ILayer& layerB) {
  CHECK(layerA.getWeights() == layerB.getWeights());
  CHECK(layerA.getBiases() == layerB.getBiases());
  CHECK(layerA.getActivationFunction().activate(10.0f) == doctest::Approx(layerB.getActivationFunction().activate(10.0f)));
}

void checkNotEqualLayers(neuro::ILayer& layerA, neuro::ILayer& layerB) {
  CHECK(layerA.getWeights() != layerB.getWeights());
  CHECK(layerA.getBiases() != layerB.getBiases());
  CHECK(layerA.getActivationFunction().activate(10.0f) != layerB.getActivationFunction().activate(10.0f));
}

TEST_CASE("DenseLayer - Initialization tests with constructors") {
  neuro::DenseLayer layer1;
  neuro::DenseLayer layer2(neuro::maker::makeRelu());

  CHECK(layer1.inputSize() == 0);
  CHECK(layer1.outputSize() == 0);

  CHECK(layer2.inputSize() == 0);
  CHECK(layer2.outputSize() == 0);

  neuro::DenseLayer layer3(1, 2);
  neuro::DenseLayer layer4(1, 2, neuro::maker::makeRelu());

  CHECK(layer3.inputSize() == 1);
  CHECK(layer3.outputSize() == 2);

  CHECK(layer4.inputSize() == 1);
  CHECK(layer4.outputSize() == 2);

  neuro::DenseLayer layer5({{0.0f, 0.0f}}, neuro::maker::makeRelu());

  CHECK(layer5.inputSize() == 2);
  CHECK(layer5.outputSize() == 1);

  neuro::DenseLayer layer6({{0.0f, 0.0f}}, {0.0f, 0.0f}, neuro::maker::makeRelu());

  CHECK(layer6.inputSize() == 2);
  CHECK(layer6.outputSize() == 1);
}

TEST_CASE("DenseLayer - Check the layer structure") {
  neuro::DenseLayer layer(2, 3, neuro::maker::makeSigmoid());

  CHECK(layer.inputSize() == 2);
  CHECK(layer.outputSize() == 3);

  layer.setBiases({0.0f, 0.0f});
  layer.setWeights({{0.0f, 0.0f, 0.0f},
                    {0.0f, 0.0f, 0.0f}});

  CHECK(layer.inputSize() == 3);
  CHECK(layer.outputSize() == 2);
}

TEST_CASE("DenseLayer - Set/Get weights and biases") {
  neuro::DenseLayer layer(1, 2, neuro::maker::makeSigmoid());

  neuro::layer_weight_t weightsComparison = {{0.75f}, {0.0f}};
  neuro::layer_bias_t biasesComparison = {0.0f, -0.25f};

  layer.setWeight(0, 0, weightsComparison[0][0]);
  layer.setBias(1, biasesComparison[1]);

  CHECK(layer.weight(0, 0) == doctest::Approx(weightsComparison[0][0]));
  CHECK(layer.bias(1) == doctest::Approx(biasesComparison[1]));

  CHECK(layer.getWeights() == weightsComparison);
  CHECK(layer.getBiases() == biasesComparison);
}

TEST_CASE("DenseLayer - Change testing via RefProxy") {
  neuro::DenseLayer layer(1, 2, neuro::maker::makeSigmoid());

  auto weight = layer.weight(1, 0);
  auto bias = layer.bias(0);

  weight = 0.75f;
  bias = 0.5f;

  CHECK(layer.weight(1, 0) == doctest::Approx(0.75f));
  CHECK(layer.bias(0) == doctest::Approx(0.5f));
}

TEST_CASE("DenseLayer - Clone") {
  neuro::DenseLayer original;

  original.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
  original.setBiases({1.0f, -1.0f});

  original.setActivationFunction(neuro::maker::makeSigmoid());

  auto clone = original.clone();

  checkEqualLayers(*clone, original);
}

TEST_CASE("DenseLayer - Copy") {
  neuro::DenseLayer original;

  original.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
  original.setBiases({1.0f, -1.0f});

  original.setActivationFunction(neuro::maker::makeSigmoid());

  neuro::DenseLayer clone(original);

  checkEqualLayers(clone, original);
}

TEST_CASE("DenseLayer - Instance attribution tests") {
  neuro::DenseLayer ref;

  ref.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
  ref.setBiases({1.0f, -1.0f});

  ref.setActivationFunction(neuro::maker::makeSigmoid());

  neuro::DenseLayer second;

  second.setWeights({{2.0f}, {4.0f}});
  second.setBiases({5.0f});

  second.setActivationFunction(neuro::maker::makeElu());

  checkNotEqualLayers(second, ref);

  second = ref;

  checkEqualLayers(second, ref);
}

TEST_CASE("DenseLayer - Reset state") {
  neuro::DenseLayer layer;

  layer.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
  layer.setBiases({1.0f, -1.0f});

  layer.reset();

  neuro::layer_weight_t weightsComparison = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  neuro::layer_bias_t biasesComparison = {0.0f, -0.0f};

  CHECK(layer.getWeights() == weightsComparison);
  CHECK(layer.getBiases() == biasesComparison);
}

TEST_CASE("DenseLayer - Randomization test of weights and biases") {
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

TEST_CASE("DenseLayer - Feedforward deterministic") {
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

TEST_CASE("DenseLayer - Index exception tests outside the range of weight and bias vectors") {
  neuro::DenseLayer layer(2, 2);

  CHECK_THROWS_AS(layer.weight(2, 2), neuro::exception::InvalidNetworkArchitectureException);
  CHECK_THROWS_AS(layer.bias(2), neuro::exception::InvalidNetworkArchitectureException);

  CHECK_THROWS_AS(layer.setWeight(2, 2, 0.0f), neuro::exception::InvalidNetworkArchitectureException);
  CHECK_THROWS_AS(layer.setBias(2, 0.0f), neuro::exception::InvalidNetworkArchitectureException);
}

TEST_CASE("DenseLayer - Index access tests within the range of weight and bias vectors") {
  neuro::DenseLayer layer(2, 2);

  CHECK_NOTHROW(layer.weight(1, 1));
  CHECK_NOTHROW(layer.bias(1));

  CHECK_NOTHROW(layer.setWeight(1, 1, 0.0f));
  CHECK_NOTHROW(layer.setBias(1, 0.0f));
}
