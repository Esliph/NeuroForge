#include "neuro/impl/neural_network.hpp"

#include <doctest/doctest.h>

#include <memory>

#include "neuro/impl/dense_layer.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/makers/activation.hpp"

template <class... Args>
std::unique_ptr<neuro::ILayer> neuralNetworkFactory(Args&&... args) {
  return std::make_unique<neuro::DenseLayer>(std::forward<Args>(args)...);
}

void checkEqualNeuralNetwork(const neuro::INeuralNetwork& networkA, const neuro::INeuralNetwork& networkB) {
  for (size_t i = 0; i < networkA.sizeLayers(); i++) {
    CHECK(networkA[i].getWeights() == networkB[i].getWeights());
    CHECK(networkA[i].getBiases() == networkB[i].getBiases());
    CHECK(networkA[i].getActivationFunction().activate(10.0f) == doctest::Approx(networkB[i].getActivationFunction().activate(10.0f)));
  }
}

void checkNotEqualNeuralNetwork(neuro::INeuralNetwork& networkA, neuro::INeuralNetwork& networkB) {
  for (size_t i = 0; i < networkA.sizeLayers(); i++) {
    CHECK(networkA[i].getWeights() != networkB[i].getWeights());
    CHECK(networkA[i].getBiases() != networkB[i].getBiases());
    CHECK(networkA[i].getActivationFunction().activate(10.0f) != doctest::Approx(networkB[i].getActivationFunction().activate(10.0f)));
  }
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork without parameters") {
  neuro::NeuralNetwork networkWithoutParameter;

  CHECK(networkWithoutParameter.sizeLayers() == 0);
  CHECK(networkWithoutParameter.inputSize() == 0);
  CHECK(networkWithoutParameter.outputSize() == 0);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by informing with the initializer") {
  neuro::DenseLayer layerSimple1(4, 3);
  neuro::DenseLayer layerSimple2(3, 2);
  neuro::DenseLayer layerSimple3(3, 2);

  neuro::NeuralNetwork networkWithInitializer = {&layerSimple1, &layerSimple2, &layerSimple3};

  CHECK(networkWithInitializer.sizeLayers() == 3);
  CHECK(networkWithInitializer.inputSize() == 4);
  CHECK(networkWithInitializer.outputSize() == 2);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by providing a list of raw layers") {
  neuro::DenseLayer layerSimple1(4, 3);
  neuro::DenseLayer layerSimple2(3, 2);
  neuro::DenseLayer layerSimple3(2, 1);

  std::vector<neuro::ILayer*> rawLayers = {&layerSimple1, &layerSimple2, &layerSimple3};

  neuro::NeuralNetwork networkWithRawLayers(rawLayers);

  CHECK(networkWithRawLayers.sizeLayers() == 3);
  CHECK(networkWithRawLayers.inputSize() == 4);
  CHECK(networkWithRawLayers.outputSize() == 1);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by providing a list of unique_ptr layers") {
  std::vector<std::unique_ptr<neuro::ILayer>> layersUniquePtr;
  layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(1, 2));
  layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(2, 3));

  neuro::NeuralNetwork networkWithLayersUniquePtr(layersUniquePtr);

  CHECK(networkWithLayersUniquePtr.sizeLayers() == 2);
  CHECK(networkWithLayersUniquePtr.inputSize() == 1);
  CHECK(networkWithLayersUniquePtr.outputSize() == 3);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by moving a list of unique_ptr layers") {
  std::vector<std::unique_ptr<neuro::ILayer>> layersUniquePtr;
  layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(1, 2));
  layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(2, 3));

  neuro::NeuralNetwork networkWithMoveLayersUniquePtr(std::move(layersUniquePtr));

  CHECK(networkWithMoveLayersUniquePtr.sizeLayers() == 2);
  CHECK(networkWithMoveLayersUniquePtr.inputSize() == 1);
  CHECK(networkWithMoveLayersUniquePtr.outputSize() == 3);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork from a list of function factories") {
  neuro::NeuralNetwork networkWithFactoryLayer({[]() { return neuralNetworkFactory(1, 2); },
                                                []() { return neuralNetworkFactory(2, 3); }});

  CHECK(networkWithFactoryLayer.sizeLayers() == 2);
  CHECK(networkWithFactoryLayer.inputSize() == 1);
  CHECK(networkWithFactoryLayer.outputSize() == 3);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork from a factory function with iteration") {
  neuro::NeuralNetwork networkWithFactoryLayer([]() { return neuralNetworkFactory(2, 2); }, 3);

  CHECK(networkWithFactoryLayer.sizeLayers() == 3);
  CHECK(networkWithFactoryLayer.inputSize() == 2);
  CHECK(networkWithFactoryLayer.outputSize() == 2);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by defining a structure of number of neurons per layer") {
  neuro::NeuralNetwork networkWithFactoryLayer({2, 4, 3, 1});

  CHECK(networkWithFactoryLayer.sizeLayers() == 3);
  CHECK(networkWithFactoryLayer.inputSize() == 2);
  CHECK(networkWithFactoryLayer.outputSize() == 1);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by defining a structure of number of neurons per layer and defining a activation function") {
  neuro::NeuralNetwork networkWithFactoryLayer({2, 4, 3, 1}, neuro::maker::makeSigmoid());

  CHECK(networkWithFactoryLayer.sizeLayers() == 3);
  CHECK(networkWithFactoryLayer.inputSize() == 2);
  CHECK(networkWithFactoryLayer.outputSize() == 1);
}

TEST_CASE("NeuralNetwork - Create NeuralNetwork by defining a structure of number of neurons per layer with multiple activation functions") {
  neuro::NeuralNetwork networkWithFactoryLayer({2, 4, 3, 1}, {neuro::maker::makeSigmoid(), neuro::maker::makeRelu(), neuro::maker::makeHard_sigmoid()});

  CHECK(networkWithFactoryLayer.sizeLayers() == 3);
  CHECK(networkWithFactoryLayer.inputSize() == 2);
  CHECK(networkWithFactoryLayer.outputSize() == 1);
}

TEST_CASE("NeuralNetwork - Check the neural network structure") {
  neuro::NeuralNetwork network({2, 4, 3, 1}, {neuro::maker::makeSigmoid(), neuro::maker::makeRelu(), neuro::maker::makeHard_sigmoid()});

  CHECK(network.sizeLayers() == 3);
  CHECK(network.inputSize() == 2);
  CHECK(network.outputSize() == 1);

  network.setLayer(0, std::make_unique<neuro::DenseLayer>(3, 4));
  network.setLayer(2, std::make_unique<neuro::DenseLayer>(3, 2));

  CHECK(network.inputSize() == 3);
  CHECK(network.outputSize() == 2);
}

TEST_CASE("NeuralNetwork - Feedforward deterministic") {
  neuro::DenseLayer layer1(neuro::maker::makeIdentity());
  layer1.setWeights({{1.0f, 2.0f},
                     {3.0f, 4.0f},
                     {5.0f, 6.0f}});
  layer1.setBiases({0.0f, 0.0f, 0.0f});

  neuro::DenseLayer layer2(neuro::maker::makeIdentity());
  layer2.setWeights({{1.0f, 1.0f, 1.0f}});
  layer2.setBiases({0.0f});

  neuro::NeuralNetwork network = {&layer1, &layer2};

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

TEST_CASE("NeuralNetwork - Add layers") {
  neuro::NeuralNetwork network;

  CHECK(network.sizeLayers() == 0);
  CHECK(network.inputSize() == 0);
  CHECK(network.outputSize() == 0);

  network.addLayer(std::make_unique<neuro::DenseLayer>(1, 2));

  CHECK(network.sizeLayers() == 1);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 2);

  neuro::DenseLayer denseLayer(2, 3);

  network.addLayer(&denseLayer);

  CHECK(network.sizeLayers() == 2);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 3);

  network.addLayer([]() { return std::make_unique<neuro::DenseLayer>(1, 2); });

  CHECK(network.sizeLayers() == 3);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 2);

  network.addLayer([]() { return std::make_unique<neuro::DenseLayer>(1, 2); }, 2);

  CHECK(network.sizeLayers() == 5);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 2);

  std::vector<std::unique_ptr<neuro::ILayer>> layers;
  layers.push_back(std::make_unique<neuro::DenseLayer>(4, 3));
  layers.push_back(std::make_unique<neuro::DenseLayer>(3, 4));

  network.addLayers(std::move(layers));

  CHECK(network.sizeLayers() == 7);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 4);
}

TEST_CASE("NeuralNetwork - Randomization test of weights and biases") {
  neuro::NeuralNetwork network({2, 2, 2}, neuro::maker::makeElu());

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

TEST_CASE("NeuralNetwork - Reset state") {
  neuro::DenseLayer layer1({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());
  neuro::DenseLayer layer2({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());
  neuro::DenseLayer layer3({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());

  neuro::NeuralNetwork network = {&layer1, &layer2, &layer3};

  CHECK(network.sizeLayers() == 3);
  CHECK(network.inputSize() == 2);
  CHECK(network.outputSize() == 2);

  network.reset();

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

TEST_CASE("NeuralNetwork - Clear layers") {
  neuro::DenseLayer layer1({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());
  neuro::DenseLayer layer2({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());
  neuro::DenseLayer layer3({{1.0f, 2.0f}, {3.0f, 4.0f}}, {1.0f, -1.0f}, neuro::maker::makeSigmoid());

  neuro::NeuralNetwork network = {&layer1, &layer2, &layer3};

  CHECK(network.sizeLayers() == 3);
  CHECK(network.inputSize() == 2);
  CHECK(network.outputSize() == 2);

  network.clearLayers();

  CHECK(network.sizeLayers() == 0);
  CHECK(network.inputSize() == 0);
  CHECK(network.outputSize() == 0);
}

TEST_CASE("NeuralNetwork - Remove layer") {
  neuro::NeuralNetwork network({1, 2, 3});

  CHECK(network.sizeLayers() == 2);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 3);

  network.removeLayer(1);

  CHECK(network.sizeLayers() == 1);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 2);
}

TEST_CASE("NeuralNetwork - Pop layer") {
  neuro::NeuralNetwork network({1, 2, 3});

  CHECK(network.sizeLayers() == 2);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 3);

  network.popLayer();

  CHECK(network.sizeLayers() == 1);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 2);
}

TEST_CASE("NeuralNetwork - Shift layer") {
  neuro::NeuralNetwork network({1, 2, 3});

  CHECK(network.sizeLayers() == 2);
  CHECK(network.inputSize() == 1);
  CHECK(network.outputSize() == 3);

  network.shiftLayer();

  CHECK(network.sizeLayers() == 1);
  CHECK(network.inputSize() == 2);
  CHECK(network.outputSize() == 3);
}

TEST_CASE("NeuralNetwork - Clone") {
  neuro::DenseLayer layerSimple1(4, 3, neuro::maker::makeIdentity());
  neuro::DenseLayer layerSimple2(3, 2, neuro::maker::makeIdentity());
  neuro::DenseLayer layerSimple3(3, 2, neuro::maker::makeIdentity());

  neuro::NeuralNetwork original = {&layerSimple1, &layerSimple2, &layerSimple3};

  auto clone = original.clone();

  checkEqualNeuralNetwork(original, *clone);
}

TEST_CASE("NeuralNetwork - Copy") {
  neuro::DenseLayer layerSimple1(4, 3, neuro::maker::makeIdentity());
  neuro::DenseLayer layerSimple2(3, 2, neuro::maker::makeIdentity());
  neuro::DenseLayer layerSimple3(3, 2, neuro::maker::makeIdentity());

  neuro::NeuralNetwork original = {&layerSimple1, &layerSimple2, &layerSimple3};

  neuro::NeuralNetwork copy(original);

  checkEqualNeuralNetwork(original, copy);
}

TEST_CASE("NeuralNetwork - Change testing via reference") {
  neuro::NeuralNetwork network({1, 2, 3}, neuro::maker::makeSigmoid());

  auto& layer = network[0];

  auto& weightProxy = layer.weight(1, 0);
  auto& biasProxy = layer.bias(0);

  weightProxy = 0.75f;
  biasProxy = 0.5f;

  CHECK(network[0].weight(1, 0) == doctest::Approx(0.75f));
  CHECK(network[0].bias(0) == doctest::Approx(0.5f));
}
