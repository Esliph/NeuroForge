#include "neuro/impl/neural_network.hpp"

#include <doctest/doctest.h>

#include <memory>

#include "interfaces/i_neural_network_test.hpp"
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
    CHECK(networkA[i].getActivationFunction().activate(10.0f) ==
          doctest::Approx(networkB[i].getActivationFunction().activate(10.0f)));
  }
}

TEST_CASE("DenseLayer - Object construction tests") {
  SUBCASE("Create NeuralNetwork without parameters") {
    neuro::NeuralNetwork networkWithoutParameter;

    CHECK(networkWithoutParameter.sizeLayers() == 0);
    CHECK(networkWithoutParameter.inputSize() == 0);
    CHECK(networkWithoutParameter.outputSize() == 0);
  }

  SUBCASE("Create NeuralNetwork by informing with the initializer") {
    neuro::DenseLayer layerSimple1(4, 3);
    neuro::DenseLayer layerSimple2(3, 2);
    neuro::DenseLayer layerSimple3(3, 2);

    neuro::NeuralNetwork networkWithInitializer = {&layerSimple1, &layerSimple2, &layerSimple3};

    CHECK(networkWithInitializer.sizeLayers() == 3);
    CHECK(networkWithInitializer.inputSize() == 4);
    CHECK(networkWithInitializer.outputSize() == 2);
  }

  SUBCASE("Create NeuralNetwork by providing a list of raw layers") {
    neuro::DenseLayer layerSimple1(4, 3);
    neuro::DenseLayer layerSimple2(3, 2);
    neuro::DenseLayer layerSimple3(2, 1);

    std::vector<neuro::ILayer*> rawLayers = {&layerSimple1, &layerSimple2, &layerSimple3};

    neuro::NeuralNetwork networkWithRawLayers(rawLayers);

    CHECK(networkWithRawLayers.sizeLayers() == 3);
    CHECK(networkWithRawLayers.inputSize() == 4);
    CHECK(networkWithRawLayers.outputSize() == 1);
  }

  SUBCASE("Create NeuralNetwork by providing a list of unique_ptr layers") {
    std::vector<std::unique_ptr<neuro::ILayer>> layersUniquePtr;
    layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(1, 2));
    layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(2, 3));

    neuro::NeuralNetwork networkWithLayersUniquePtr(layersUniquePtr);

    CHECK(networkWithLayersUniquePtr.sizeLayers() == 2);
    CHECK(networkWithLayersUniquePtr.inputSize() == 1);
    CHECK(networkWithLayersUniquePtr.outputSize() == 3);
  }

  SUBCASE("Create NeuralNetwork by moving a list of unique_ptr layers") {
    std::vector<std::unique_ptr<neuro::ILayer>> layersUniquePtr;
    layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(1, 2));
    layersUniquePtr.push_back(std::make_unique<neuro::DenseLayer>(2, 3));

    neuro::NeuralNetwork networkWithMoveLayersUniquePtr(std::move(layersUniquePtr));

    CHECK(networkWithMoveLayersUniquePtr.sizeLayers() == 2);
    CHECK(networkWithMoveLayersUniquePtr.inputSize() == 1);
    CHECK(networkWithMoveLayersUniquePtr.outputSize() == 3);
  }

  SUBCASE("Create NeuralNetwork from a list of function factories") {
    neuro::NeuralNetwork networkWithFactoryLayer(
      {[]() { return neuralNetworkFactory(1, 2); }, []() { return neuralNetworkFactory(2, 3); }});

    CHECK(networkWithFactoryLayer.sizeLayers() == 2);
    CHECK(networkWithFactoryLayer.inputSize() == 1);
    CHECK(networkWithFactoryLayer.outputSize() == 3);
  }

  SUBCASE("Create NeuralNetwork from a factory function with iteration") {
    neuro::NeuralNetwork networkWithFactoryLayer([]() { return neuralNetworkFactory(2, 2); }, 3);

    CHECK(networkWithFactoryLayer.sizeLayers() == 3);
    CHECK(networkWithFactoryLayer.inputSize() == 2);
    CHECK(networkWithFactoryLayer.outputSize() == 2);
  }

  SUBCASE("Create NeuralNetwork by defining a structure of number of neurons per layer") {
    neuro::NeuralNetwork networkWithFactoryLayer({2, 4, 3, 1});

    CHECK(networkWithFactoryLayer.sizeLayers() == 3);
    CHECK(networkWithFactoryLayer.inputSize() == 2);
    CHECK(networkWithFactoryLayer.outputSize() == 1);
  }

  SUBCASE("Create NeuralNetwork by defining a structure of number of neurons per layer and defining a activation function") {
    neuro::NeuralNetwork networkWithFactoryLayer({2, 4, 3, 1}, neuro::maker::activationSigmoid());

    CHECK(networkWithFactoryLayer.sizeLayers() == 3);
    CHECK(networkWithFactoryLayer.inputSize() == 2);
    CHECK(networkWithFactoryLayer.outputSize() == 1);
  }

  SUBCASE("Create a NeuralNetwork from a prototype") {
    neuro::DenseLayer prototype(2, 2);

    neuro::NeuralNetwork networkWithPrototype(prototype, 3);

    CHECK(networkWithPrototype.sizeLayers() == 3);
    CHECK(networkWithPrototype.inputSize() == 2);
    CHECK(networkWithPrototype.outputSize() == 2);
  }
}

TEST_IMPL_INEURAL_NETWORK("NeuralNetwork", neuro::NeuralNetwork);
