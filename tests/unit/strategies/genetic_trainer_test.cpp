#include "neuro/strategies/genetic_trainer.hpp"

#include <doctest/doctest.h>

#include <iostream>
#include <memory>
#include <vector>

#include "neuro/impl/population.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/makers/activation.hpp"

TEST_CASE("Teste") {
  std::vector<int> structure = {1, 1};
  std::shared_ptr<neuro::IPopulation> population = std::make_shared<neuro::Population>(2, structure);

  neuro::GeneticOptions options;
  options.rate = 1.0f;

  neuro::GeneticTrainer trainer(std::move(population), options);

  trainer.mutate();

  for (const auto& individual : *population) {
    CHECK(individual->getNeuralNetwork().layer(0).getWeight(0, 0) != doctest::Approx(0.0f));
  }
}
