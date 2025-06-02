#include "neuro/population.hpp"

#include <vector>

#include "neuro/activation.hpp"
#include "neuro/individual.hpp"
#include "neuro/util.hpp"

namespace neuro {

Population::Population(const std::vector<Individual>& individuals)
    : individuals(individuals) {}

Population::Population(int size, const std::vector<int>& structure, const ActivationFunction& activation) {
  for (size_t i = 0; i < size; ++i) {
    individuals.emplace_back(structure, activation);
  }
}

Population::Population(int size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations) {
  for (size_t i = 0; i < size; ++i) {
    individuals.emplace_back(structure, activations);
  }
}

void Population::loadWeights(float rangeMin, float rangeMax) {
  for (size_t i = 0; i < individuals.size(); ++i) {
    individuals[i].loadWeights(rangeMin, rangeMax);
  }
}

void Population::loadBias(float rangeMin, float rangeMax) {
  for (size_t i = 0; i < individuals.size(); ++i) {
    individuals[i].loadBias(rangeMin, rangeMax);
  }
}

const Individual& Population::getBest() const {
  return *std::max_element(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
    return a.getFitness() > b.getFitness();
  });
}

void Population::evolve(float mutationRate, float mutationStrength, unsigned int eliteCount) {
  std::sort(individuals.begin(), individuals.end(), [](const Individual& a, const Individual& b) {
    return a.getFitness() > b.getFitness();
  });

  std::vector<Individual> nextGeneration;

  for (size_t i = 0; i < eliteCount && i < individuals.size(); ++i) {
    nextGeneration.emplace_back(individuals[i]);
  }

  while (nextGeneration.size() < individuals.size()) {
    const Individual& parent1 = tournamentSelect();
    const Individual& parent2 = tournamentSelect();

    Individual child = parent1.crossover(parent2);

    child.mutate(mutationRate, mutationStrength);
    nextGeneration.push_back(child);
  }

  individuals = std::move(nextGeneration);
}

const Individual& Population::tournamentSelect(int rounds) const {
  std::uniform_int_distribution<int> dist(0, individuals.size() - 1);

  const Individual* winner = nullptr;

  for (int i = 0; i < rounds; ++i) {
    const Individual& candidate = individuals[dist(random_engine)];

    if (!winner || candidate.getFitness() > winner->getFitness()) {
      winner = &candidate;
    }
  }

  return *winner;
}

std::vector<Individual>::const_iterator Population::begin() const {
  return individuals.begin();
}

std::vector<Individual>::const_iterator Population::end() const {
  return individuals.end();
}

std::vector<Individual>::iterator Population::begin() {
  return individuals.begin();
}

std::vector<Individual>::iterator Population::end() {
  return individuals.end();
}

std::vector<Individual>& Population::getIndividuals() {
  return individuals;
}

};  // namespace neuro
