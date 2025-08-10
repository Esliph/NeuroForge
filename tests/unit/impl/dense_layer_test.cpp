#include "neuro/impl/dense_layer.hpp"

#include <doctest/doctest.h>

#include "interfaces/i_layer_test.hpp"
#include "neuro/makers/activation.hpp"

TEST_CASE("DenseLayer - Object construction tests") {
  neuro::layer_weight_t weights = {{3.0f, 6.0f}};
  neuro::layer_bias_t biases = {3.0f, 6.0f};

  SUBCASE("Create DenseLayer without parameters") {
    neuro::DenseLayer layerWithoutParameter;

    CHECK(layerWithoutParameter.inputSize() == 0);
    CHECK(layerWithoutParameter.outputSize() == 0);
  }

  SUBCASE("Create DenseLayer informing only the activation function") {
    neuro::DenseLayer layerWithOnlyActivation(neuro::maker::activationRelu());

    CHECK(layerWithOnlyActivation.inputSize() == 0);
    CHECK(layerWithOnlyActivation.outputSize() == 0);
  }

  SUBCASE("Create DenseLayer by specifying input and output size") {
    neuro::DenseLayer layerWithInputAndOutput(1, 2);
    neuro::DenseLayer layerWithInputAndOutputAndActivation(1, 2, neuro::maker::activationRelu());

    CHECK(layerWithInputAndOutput.inputSize() == 1);
    CHECK(layerWithInputAndOutput.outputSize() == 2);

    CHECK(layerWithInputAndOutputAndActivation.inputSize() == 1);
    CHECK(layerWithInputAndOutputAndActivation.outputSize() == 2);
  }

  SUBCASE("Create DenseLayer informing only the weights") {
    neuro::DenseLayer layerWithWeights(weights);

    CHECK(layerWithWeights.inputSize() == 2);
    CHECK(layerWithWeights.outputSize() == 1);
    CHECK(layerWithWeights.getWeights() == weights);
  }

  SUBCASE("Create DenseLayer informing only the biases") {
    neuro::DenseLayer layerWithBiases(biases);

    CHECK(layerWithBiases.inputSize() == 0);
    CHECK(layerWithBiases.outputSize() == 2);
    CHECK(layerWithBiases.getBiases() == biases);
  }

  SUBCASE("Create DenseLayer informing the weights and activation function") {
    neuro::DenseLayer layerWithWeightsAndActivation(weights, neuro::maker::activationRelu());

    CHECK(layerWithWeightsAndActivation.inputSize() == 2);
    CHECK(layerWithWeightsAndActivation.outputSize() == 1);
    CHECK(layerWithWeightsAndActivation.getWeights() == weights);
  }

  SUBCASE("Create DenseLayer informing the biases and activation function") {
    neuro::DenseLayer layerWithBiasesAndActivation(biases, neuro::maker::activationRelu());

    CHECK(layerWithBiasesAndActivation.inputSize() == 0);
    CHECK(layerWithBiasesAndActivation.outputSize() == 2);
    CHECK(layerWithBiasesAndActivation.getBiases() == biases);
  }

  SUBCASE("Create DenseLayer informing the weights and biases") {
    neuro::DenseLayer layerWithWeightsAndBiases(weights, biases);

    CHECK(layerWithWeightsAndBiases.inputSize() == 2);
    CHECK(layerWithWeightsAndBiases.outputSize() == 1);
    CHECK(layerWithWeightsAndBiases.getWeights() == weights);
    CHECK(layerWithWeightsAndBiases.getBiases() == biases);
  }

  SUBCASE("Create DenseLayer informing the weights, biases and activation function") {
    neuro::DenseLayer layerWithWeightsAndBiasesAndActivation(weights, biases, neuro::maker::activationRelu());

    CHECK(layerWithWeightsAndBiasesAndActivation.inputSize() == 2);
    CHECK(layerWithWeightsAndBiasesAndActivation.outputSize() == 1);
    CHECK(layerWithWeightsAndBiasesAndActivation.getWeights() == weights);
    CHECK(layerWithWeightsAndBiasesAndActivation.getBiases() == biases);
  }
}

TEST_IMPL_ILAYER("DenseLayer", neuro::DenseLayer);
