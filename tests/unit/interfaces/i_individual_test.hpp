#include <doctest/doctest.h>

#include <memory>

#include "neuro/impl/dense_layer.hpp"
#include "neuro/impl/neural_network.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

#define TEST_IMPL_IINDIVIDUAL(NAME, TYPE)                                 \
  TEST_CASE(NAME " - Testing implementation for IIndividual interface") { \
    runTestInterfaceIIndividual<TYPE>();                                  \
  }

template <typename IIndividualImpl>
void runTestInterfaceIIndividual() {
  SUBCASE("Testing changes to the Neural Network object") {
    IIndividualImpl individual;

    CHECK(individual.getNeuralNetwork().sizeLayers() == 0);
    CHECK(individual.getNeuralNetwork().inputSize() == 0);
    CHECK(individual.getNeuralNetwork().outputSize() == 0);

    neuro::NeuralNetwork neuralNetwork({1, 2, 3});

    individual.setNeuralNetwork(neuralNetwork);

    CHECK(individual.getNeuralNetwork().sizeLayers() == 2);
    CHECK(individual.getNeuralNetwork().inputSize() == 1);
    CHECK(individual.getNeuralNetwork().outputSize() == 3);

    std::vector<int> structure = {2, 4, 4, 2};
    std::unique_ptr<neuro::INeuralNetwork> neuralNetworkPtr = std::make_unique<neuro::NeuralNetwork>(structure);

    individual.setNeuralNetwork(std::move(neuralNetworkPtr));

    CHECK(individual.getNeuralNetwork().sizeLayers() == 3);
    CHECK(individual.getNeuralNetwork().inputSize() == 2);
    CHECK(individual.getNeuralNetwork().outputSize() == 2);
  }

  SUBCASE("Testing changes in individual fitness") {
    IIndividualImpl individual;

    CHECK(individual.getFitness() == doctest::Approx(0.0f));

    individual.setFitness(10.0f);

    CHECK(individual.getFitness() == doctest::Approx(10.0f));
  }

  SUBCASE("Testing the application of fitness in individual assessment") {
    IIndividualImpl individual;

    CHECK(individual.getFitness() == doctest::Approx(0.0f));

    individual.evaluateFitness([]([[maybe_unused]] const neuro::INeuralNetwork& network) {
      return 10.0f;
    });

    CHECK(individual.getFitness() == doctest::Approx(10.0f));
  }

  SUBCASE("Clone") {
    IIndividualImpl original;

    neuro::NeuralNetwork neuralNetwork({1, 2, 3});

    original.setNeuralNetwork(neuralNetwork);

    auto copy = original.clone();

    CHECK(copy->getNeuralNetwork().sizeLayers() == 2);
    CHECK(copy->getNeuralNetwork().inputSize() == 1);
    CHECK(copy->getNeuralNetwork().outputSize() == 3);
  }
}
