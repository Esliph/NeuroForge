#include <chrono>
#include <random>

#include "common/measure.hpp"
#include "neuro/neuro.hpp"

void testEvolvePopulation() {
  std::default_random_engine engine((std::random_device())());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  neuro::GeneticTrainer trainer;

  neuro::Population pop(100000, {2, 3, 4, 3}, neuro::maker::activationSigmoid());

  for (auto& individual : pop) {
    individual->setFitness(dist(engine));
  }

  neuro::Measure::run("Train population", [&trainer, &pop]() { trainer.evolve(pop); }, 3);
}

int main() {
  testEvolvePopulation();

  return 0;
}
