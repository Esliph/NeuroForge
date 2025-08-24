#pragma once

#include <doctest/doctest.h>

#include <functional>

#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/makers/activation.hpp"
#include "neuro/types.hpp"

#define TEST_IMPL_ILAYER(NAME, TYPE)                                 \
  TEST_CASE(NAME " - Testing implementation for ILayer interface") { \
    runTestInterfaceILayer<TYPE>();                                  \
  }

void checkEqualsILayer(neuro::ILayer& layerA, neuro::ILayer& layerB) {
  CHECK(layerA.getWeights() == layerB.getWeights());
  CHECK(layerA.getBiases() == layerB.getBiases());
  CHECK(layerA.getActivationFunction().activate(10.0f) == doctest::Approx(layerB.getActivationFunction().activate(10.0f)));
}

void checkNotEqualsILayer(neuro::ILayer& layerA, neuro::ILayer& layerB) {
  CHECK(layerA.getWeights() != layerB.getWeights());
  CHECK(layerA.getBiases() != layerB.getBiases());
  CHECK(layerA.getActivationFunction().activate(10.0f) != layerB.getActivationFunction().activate(10.0f));
}

template <typename ILayerImpl>
void runTestInterfaceILayer() {
  SUBCASE("Check the layer structure") {
    ILayerImpl layer;

    CHECK(layer.inputSize() == 0);
    CHECK(layer.outputSize() == 0);

    layer.setBiases({0.0f, 0.0f});
    layer.setWeights({{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}});

    CHECK(layer.inputSize() == 3);
    CHECK(layer.outputSize() == 2);
  }

  SUBCASE("Set/Get weights and biases") {
    ILayerImpl layer;

    layer.setBiases({0.0f, 0.0f});
    layer.setWeights({{0.0f}, {0.0f}});

    neuro::layer_weight_t weightsComparison = {{0.75f}, {0.0f}};
    neuro::layer_bias_t biasesComparison = {0.0f, -0.25f};

    layer.setWeight(0, 0, weightsComparison[0][0]);
    layer.setBias(1, biasesComparison[1]);

    CHECK(layer.getWeight(0, 0) == doctest::Approx(weightsComparison[0][0]));
    CHECK(layer.getBias(1) == doctest::Approx(biasesComparison[1]));

    CHECK(layer.weightRef(0, 0) == doctest::Approx(weightsComparison[0][0]));
    CHECK(layer.biasRef(1) == doctest::Approx(biasesComparison[1]));

    CHECK(layer.getWeights() == weightsComparison);
    CHECK(layer.getBiases() == biasesComparison);
  }

  SUBCASE("Change testing via reference") {
    ILayerImpl layer;

    layer.setBiases({0.0f, 0.0f});
    layer.setWeights({{0.0f}, {0.0f}});

    layer.setActivationFunction(neuro::maker::activationSigmoid());

    auto& weight = layer.weightRef(1, 0);
    auto& bias = layer.biasRef(0);

    weight = 0.75f;
    bias = 0.5f;

    CHECK(layer.weightRef(1, 0) == doctest::Approx(0.75f));
    CHECK(layer.biasRef(0) == doctest::Approx(0.5f));
  }

  SUBCASE("Clone") {
    ILayerImpl original;

    original.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
    original.setBiases({1.0f, -1.0f});

    original.setActivationFunction(neuro::maker::activationSigmoid());

    auto clone = original.clone();

    checkEqualsILayer(*clone, original);
  }

  SUBCASE("Copy") {
    ILayerImpl original;

    original.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
    original.setBiases({1.0f, -1.0f});

    original.setActivationFunction(neuro::maker::activationSigmoid());

    ILayerImpl clone(original);

    checkEqualsILayer(clone, original);
  }

  SUBCASE("Reset state") {
    neuro::layer_weight_t weightsComparison = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    neuro::layer_bias_t biasesComparison = {0.0f, -0.0f};

    SUBCASE("Resetting layer state without changing the structure") {
      ILayerImpl layer;

      layer.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
      layer.setBiases({1.0f, -1.0f});

      layer.clear();

      CHECK(layer.getWeights() == weightsComparison);
      CHECK(layer.getBiases() == biasesComparison);
    }

    SUBCASE("Resetting layer state and defining a new structure") {
      ILayerImpl layer;

      layer.setWeights({{1.0f, 2.0f, 0.0f}, {3.0f, 4.0f, 0.0f}, {5.0f, 6.0f, 0.0f}});
      layer.setBiases({1.0f, -1.0f, 0.0f});

      layer.reshape(2, 2);

      CHECK(layer.getWeights() == weightsComparison);
      CHECK(layer.getBiases() == biasesComparison);
    }
  }

  SUBCASE("Instance attribution tests") {
    ILayerImpl ref;

    ref.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
    ref.setBiases({1.0f, -1.0f});

    ref.setActivationFunction(neuro::maker::activationSigmoid());

    ILayerImpl second;

    second.setWeights({{2.0f}, {4.0f}});
    second.setBiases({5.0f});

    second.setActivationFunction(neuro::maker::activationElu());

    checkNotEqualsILayer(second, ref);

    second = ref;

    checkEqualsILayer(second, ref);
  }

  SUBCASE("Randomization test of weights and biases") {
    ILayerImpl layer;

    layer.reshape(4, 4);

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

  SUBCASE("Change in weights by mutation methods") {
    ILayerImpl layer;

    layer.reshape(1, 1);
    layer.setWeight(0, 0, 1.0f);

    layer.mutateWeights([&](float weight) {
      CHECK(weight == 1.0f);

      return 10.0f;
    });

    CHECK(layer.getWeight(0, 0) == 10.0f);
  }

  SUBCASE("Change in biases by mutation methods") {
    ILayerImpl layer;

    layer.reshape(1, 1);
    layer.setBias(0, 1.0f);

    layer.mutateBiases([&](float bias) {
      CHECK(bias == 1.0f);

      return 10.0f;
    });

    CHECK(layer.getBias(0) == 10.0f);
  }

  SUBCASE("Feedforward deterministic") {
    ILayerImpl layer;

    layer.setWeights({{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});
    layer.setBiases({0.5f, -0.5f});

    layer.setActivationFunction(neuro::maker::activationRelu());

    neuro::neuro_layer_t input = {1.0f, 2.0f, 3.0f};
    auto output = layer.feedforward(input);

    CHECK(output.size() == 2);
    CHECK(output[0] == doctest::Approx(0.1f * 1 + 0.2f * 2 + 0.3f * 3 + 0.5f));
    CHECK(output[1] == doctest::Approx(std::max(0.0f, 0.4f * 1 + 0.5f * 2 + 0.6f * 3 - 0.5f)));
  }

  SUBCASE("Index exception tests outside the range of weight and bias vectors") {
    ILayerImpl layer;

    layer.setWeights({{0.0f, 0.0f}, {0.0f, 0.0f}});
    layer.setBiases({0.0f, 0.0f});

    CHECK_THROWS_AS(layer.weightRef(2, 2), neuro::exception::InvalidNetworkArchitectureException);
    CHECK_THROWS_AS(layer.biasRef(2), neuro::exception::InvalidNetworkArchitectureException);

    CHECK_THROWS_AS(layer.setWeight(2, 2, 0.0f), neuro::exception::InvalidNetworkArchitectureException);
    CHECK_THROWS_AS(layer.setBias(2, 0.0f), neuro::exception::InvalidNetworkArchitectureException);
  }

  SUBCASE("Index access tests within the range of weight and bias vectors") {
    ILayerImpl layer;

    layer.setWeights({{0.0f, 0.0f}, {0.0f, 0.0f}});
    layer.setBiases({0.0f, 0.0f});

    CHECK_NOTHROW(layer.weightRef(1, 1));
    CHECK_NOTHROW(layer.biasRef(1));

    CHECK_NOTHROW(layer.setWeight(1, 1, 0.0f));
    CHECK_NOTHROW(layer.setBias(1, 0.0f));
  }

  SUBCASE("Validating the arithmetic mean of layer weights") {
    ILayerImpl layer;

    neuro::layer_weight_t weights = {{10.0f, -5.0f, 8.0f}, {3.0f, -2.0f, 12.0f}, {6.0f, -3.0f, 15.0f}};

    layer.setWeights(weights);

    float sum = 0;

    for (size_t i = 0; i < weights.size(); i++) {
      for (size_t j = 0; j < weights[i].size(); j++) {
        sum += weights[i][j];
      }
    }

    CHECK(layer.meanWeight() == doctest::Approx(sum / (weights.size() * weights[0].size())));
  }

  SUBCASE("Validating the arithmetic mean of layer biases") {
    ILayerImpl layer;

    neuro::layer_bias_t biases = {10.0f, -5.0f, 8.0f};

    layer.setBiases(biases);

    float sum = 0;

    for (size_t i = 0; i < biases.size(); i++) {
      sum += biases[i];
    }

    CHECK(layer.meanBias() == doctest::Approx(sum / biases.size()));
  }

  SUBCASE("Testing the internal structure of the layer") {
    SUBCASE("Correct inner layer") {
      ILayerImpl layer;

      layer.setWeights({{1.0f, 2.0f}, {3.0f, 4.0f}});
      layer.setBiases({1.0f, -1.0f});

      CHECK(layer.validateInternalShape());
    }

    SUBCASE("Incorrect inner layer") {
      ILayerImpl layer;

      layer.setWeights({{1.0f, 3.0f}});
      layer.setBiases({1.0f, -1.0f});

      CHECK(!layer.validateInternalShape());
    }

    SUBCASE("Layer with weight matrix with different outputs") {
      ILayerImpl layer;

      layer.setWeights({{1.0f}, {1.0f, 3.0f}});
      layer.setBiases({1.0f, -1.0f});

      CHECK(!layer.validateInternalShape());

      layer.setWeights({{1.0f, 3.0f}, {1.0f}});
      layer.setBiases({1.0f, -1.0f});

      CHECK(!layer.validateInternalShape());
    }
  }
}
