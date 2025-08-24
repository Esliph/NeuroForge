#include "neuro/strategies/genetic_trainer.hpp"

#include <doctest/doctest.h>

#include <iostream>
#include <memory>
#include <vector>

#include "neuro/impl/population.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/makers/activation.hpp"

TEST_CASE("GeneticTrainer - Testing changes to option parameters") {
  std::vector<int> structure = {1, 1};
  std::shared_ptr<neuro::IPopulation> population = std::make_shared<neuro::Population>(2, structure);

  const neuro::GeneticOptions defaultOptions{};

  neuro::GeneticTrainer trainer(population);

  REQUIRE(trainer.getOptions() == defaultOptions);

  SUBCASE("Rate change tests") {
    trainer.setRate(0.75f);

    CHECK(trainer.getOptions().rate == doctest::Approx(0.75f));
  }

  SUBCASE("Intensity change tests") {
    trainer.setIntensity(0.25f);

    CHECK(trainer.getOptions().intensity == doctest::Approx(0.25f));
  }

  SUBCASE("Elite Count change tests") {
    trainer.setEliteCount(10);

    CHECK(trainer.getOptions().eliteCount == 10);
  }
}

TEST_CASE("GeneticTrainer - Testing changes in weights and biases through mutation") {
  std::vector<int> structure = {1, 1};
  std::shared_ptr<neuro::IPopulation> population = std::make_shared<neuro::Population>(2, structure);

  neuro::GeneticOptions options;
  options.rate = 1.0f;

  neuro::GeneticTrainer trainer(std::move(population), options);

  trainer.mutate();

  for (const auto& individual : *population) {
    const auto& network = individual->getNeuralNetwork();

    for (const auto& layer : network) {
      const auto& weights = layer->getWeights();

      for (const auto& lines : weights) {
        for (const auto& line : lines) {
          CHECK(line != doctest::Approx(0.0f));
        }
      }

      const auto& biases = layer->getBiases();

      for (const auto& bias : biases) {
        CHECK(bias != doctest::Approx(0.0f));
      }
    }
  }
}
