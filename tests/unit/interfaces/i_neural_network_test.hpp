#include <doctest/doctest.h>

#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"
#include "neuro/makers/activation.hpp"

#define TEST_IMPL_INEURAL_NETWORK(NAME, TYPE)                                \
  TEST_CASE(NAME " - Testing implementation for INeuralNetwork interface") { \
    runTestInterfaceINeuralNetwork<TYPE>();                                  \
  }

void checkEqualsINeuralNetwork(const neuro::INeuralNetwork& networkA, const neuro::INeuralNetwork& networkB) {
  REQUIRE(networkA.sizeLayers() == networkB.sizeLayers());

  for (size_t i = 0; i < networkA.sizeLayers(); i++) {
    CHECK(networkA[i].getWeights() == networkB[i].getWeights());
    CHECK(networkA[i].getBiases() == networkB[i].getBiases());
    CHECK(networkA[i].getActivationFunction().activate(10.0f) == doctest::Approx(networkB[i].getActivationFunction().activate(10.0f)));
  }
}

template <typename INeuralNetworkImpl>
void runTestInterfaceINeuralNetwork() {
  SUBCASE("Check the neural network structure") {
    INeuralNetworkImpl network;

    network.restructure({2, 4, 3, 1});

    CHECK(network.sizeLayers() == 3);
    CHECK(network.inputSize() == 2);
    CHECK(network.outputSize() == 1);

    network.setLayer(0, std::make_unique<neuro::DenseLayer>(3, 4));
    network.setLayer(2, std::make_unique<neuro::DenseLayer>(3, 2));

    CHECK(network.inputSize() == 3);
    CHECK(network.outputSize() == 2);
  }

  SUBCASE("Feedforward deterministic") {
    neuro::layer_weight_t weights1 = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    neuro::layer_bias_t biases1 = {0.0f, 0.0f, 0.0f};

    neuro::layer_weight_t weights2 = {{1.0f, 1.0f, 1.0f}};
    neuro::layer_bias_t biases2 = {0.0f};

    std::vector<std::unique_ptr<neuro::ILayer>> layers;

    layers.push_back(std::make_unique<neuro::DenseLayer>(weights1, biases1, neuro::maker::activationIdentity()));
    layers.push_back(std::make_unique<neuro::DenseLayer>(weights2, biases2, neuro::maker::activationIdentity()));

    INeuralNetworkImpl network;

    network.setLayers(std::move(layers));

    neuro::neuro_layer_t input = {1.0f, 2.0f};

    // Feedforward manual:
    // Layer 1:
    //   Neuron 1: 1 * 1 + 2 * 2 = 5
    //   Neuron 2: 1 * 3 + 2 * 4 = 11
    //   Neuron 3: 1 * 5 + 2 * 6 = 17
    // Layer 2:
    //   Neuron 1: 5 + 11 + 17 = 33

    auto output = network.feedforward(input);

    REQUIRE(output.size() == 1);

    CHECK(output[0] == doctest::Approx(33.0f));
    CHECK(network(input) == output);
  }

  SUBCASE("Add layers") {
    INeuralNetworkImpl network;

    CHECK(network.sizeLayers() == 0);
    CHECK(network.inputSize() == 0);
    CHECK(network.outputSize() == 0);
    CHECK(network.empty());

    network.addLayer(std::make_unique<neuro::DenseLayer>(1, 2));

    CHECK(network.sizeLayers() == 1);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 2);
    CHECK(!network.empty());

    network.addLayer([]() { return std::make_unique<neuro::DenseLayer>(1, 3); });

    CHECK(network.sizeLayers() == 2);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 3);

    network.addLayer([]() { return std::make_unique<neuro::DenseLayer>(1, 2); }, 2);

    CHECK(network.sizeLayers() == 4);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 2);

    std::vector<std::unique_ptr<neuro::ILayer>> layers;
    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3));
    layers.push_back(std::make_unique<neuro::DenseLayer>(3, 4));

    network.addLayers(std::move(layers));

    CHECK(network.sizeLayers() == 6);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 4);
  }

  SUBCASE("Clear layers") {
    std::vector<std::unique_ptr<neuro::ILayer>> layers;

    neuro::layer_weight_t weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    neuro::layer_bias_t biases = {1.0f, -1.0f};

    layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
    layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
    layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));

    INeuralNetworkImpl network;

    network.setLayers(std::move(layers));

    CHECK(network.sizeLayers() == 3);
    CHECK(network.inputSize() == 2);
    CHECK(network.outputSize() == 2);

    network.clearLayers();

    CHECK(network.sizeLayers() == 0);
    CHECK(network.inputSize() == 0);
    CHECK(network.outputSize() == 0);
  }

  SUBCASE("Empty layers") {
    INeuralNetworkImpl network;

    CHECK(network.empty());

    network.restructure({1, 2});

    CHECK(!network.empty());
  }

  SUBCASE("Remove layer") {
    INeuralNetworkImpl network;

    network.restructure({1, 2, 3});

    CHECK(network.sizeLayers() == 2);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 3);

    network.removeLayer(1);

    CHECK(network.sizeLayers() == 1);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 2);
  }

  SUBCASE("Pop layer") {
    INeuralNetworkImpl network;

    network.restructure({1, 2, 3});

    CHECK(network.sizeLayers() == 2);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 3);

    network.popLayer();

    CHECK(network.sizeLayers() == 1);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 2);
  }

  SUBCASE("Shift layer") {
    INeuralNetworkImpl network;

    network.restructure({1, 2, 3});

    CHECK(network.sizeLayers() == 2);
    CHECK(network.inputSize() == 1);
    CHECK(network.outputSize() == 3);

    network.shiftLayer();

    CHECK(network.sizeLayers() == 1);
    CHECK(network.inputSize() == 2);
    CHECK(network.outputSize() == 3);
  }

  SUBCASE("Randomization test of weights and biases") {
    INeuralNetworkImpl network;

    network.restructure({2, 2, 2});

    network.randomizeWeights(-1.0f, 1.0f);
    network.randomizeBiases(-2.0f, 2.0f);

    for (const auto& layer : network) {
      for (const auto& neurons : layer->getWeights()) {
        for (float weight : neurons) {
          CHECK(weight >= -1.0f);
          CHECK(weight <= 1.0f);
        }
      }

      for (const float bias : layer->getBiases()) {
        CHECK(bias >= -2.0f);
        CHECK(bias <= 2.0f);
      }
    }
  }

  SUBCASE("Clone") {
    std::vector<std::unique_ptr<neuro::ILayer>> layers;

    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));
    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));
    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));

    INeuralNetworkImpl original;

    original.setLayers(std::move(layers));

    auto clone = original.clone();

    checkEqualsINeuralNetwork(original, *clone);
  }

  SUBCASE("Copy") {
    std::vector<std::unique_ptr<neuro::ILayer>> layers;

    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));
    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));
    layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3, neuro::maker::activationIdentity()));

    INeuralNetworkImpl original;

    original.setLayers(std::move(layers));

    INeuralNetworkImpl copy(original);

    checkEqualsINeuralNetwork(original, copy);
  }

  SUBCASE("Change testing via reference") {
    INeuralNetworkImpl network;

    network.restructure({1, 2, 3});

    auto& layer = network[0];

    auto& weight = layer.weightRef(1, 0);
    auto& bias = layer.biasRef(0);

    weight = 0.75f;
    bias = 0.5f;

    CHECK(network[0].weightRef(1, 0) == doctest::Approx(0.75f));
    CHECK(network[0].biasRef(0) == doctest::Approx(0.5f));
  }

  SUBCASE("Reset state") {
    SUBCASE("Resetting neural network state without changing the structure") {
      std::vector<std::unique_ptr<neuro::ILayer>> layers;

      neuro::layer_weight_t weights = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      neuro::layer_bias_t biases = {1.0f, -1.0f};

      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));

      INeuralNetworkImpl network;

      network.setLayers(std::move(layers));

      CHECK(network.sizeLayers() == 3);
      CHECK(network.inputSize() == 2);
      CHECK(network.outputSize() == 2);

      network.clear();

      CHECK(network.sizeLayers() == 3);
      CHECK(network.inputSize() == 2);
      CHECK(network.outputSize() == 2);

      neuro::layer_weight_t weightsComparison = {{0.0f, 0.0f}, {0.0f, 0.0f}};
      neuro::layer_bias_t biasesComparison = {0.0f, -0.0f};

      for (const auto& layer : network) {
        CHECK(layer->getWeights() == weightsComparison);
        CHECK(layer->getBiases() == biasesComparison);
      }
    }

    SUBCASE("Resetting neural network state and defining a new structure") {
      std::vector<std::unique_ptr<neuro::ILayer>> layers;

      neuro::layer_weight_t weights = {{1.0f, 2.0f}, {3.0f, 4.0f, 2.0f}, {5.0f, 6.0f, 2.0f}};
      neuro::layer_bias_t biases = {1.0f, -1.0f, 0.0f};

      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));
      layers.push_back(std::make_unique<neuro::DenseLayer>(weights, biases));

      INeuralNetworkImpl network;

      network.setLayers(std::move(layers));

      CHECK(network.sizeLayers() == 4);
      CHECK(network.inputSize() == 2);
      CHECK(network.outputSize() == 3);

      network.restructure({2, 2, 2});

      CHECK(network.sizeLayers() == 2);
      CHECK(network.inputSize() == 2);
      CHECK(network.outputSize() == 2);

      neuro::layer_weight_t weightsComparison = {{0.0f, 0.0f}, {0.0f, 0.0f}};
      neuro::layer_bias_t biasesComparison = {0.0f, -0.0f};

      for (const auto& layer : network) {
        CHECK(layer->getWeights() == weightsComparison);
        CHECK(layer->getBiases() == biasesComparison);
      }
    }
  }
}
