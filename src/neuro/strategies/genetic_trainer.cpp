#include "neuro/strategies/genetic_trainer.hpp"

#include <memory>

#include "internal/random_engine.hpp"
#include "neuro/interfaces/i_population.hpp"

namespace neuro {

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population)
    : population(population) {}

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population, float rate, float intensity, size_t eliteCount)
    : population(population),
      options({rate, intensity, eliteCount}) {}

  GeneticTrainer::GeneticTrainer(const std::shared_ptr<IPopulation>& population, const GeneticOptions& options)
    : population(population),
      options(options) {}

  void GeneticTrainer::evolve() {
  }

  void GeneticTrainer::mutate() {
    auto& individuals = population->getIndividuals();

    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::normal_distribution<float> perturbation(-options.intensity, options.intensity);

    for (auto& individual : individuals) {
      auto& network = individual->getNeuralNetwork();

      network.mutateWeights([&](float) {
        return chance(random_engine) < options.rate ? perturbation(random_engine) : 0.0f;
      });

      network.mutateBiases([&](float) {
        return chance(random_engine) < options.rate ? perturbation(random_engine) : 0.0f;
      });
    }
  }

  void GeneticTrainer::crossover() {
  }

}; // namespace neuro
