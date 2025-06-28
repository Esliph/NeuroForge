#include "neuro/strategies/genetic_trainer.hpp"

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "neuro/impl/population.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_layer.hpp"
#include "neuro/utils/random_engine.hpp"

namespace neuro {

GeneticTrainer::GeneticTrainer(float rate, float intensity, size_t eliteCount)
    : options({rate, intensity, eliteCount}) {}

GeneticTrainer::GeneticTrainer(const GeneticOptions& options)
    : options(options) {}

void GeneticTrainer::mutateWeights(layer_weight_t& layerWeights) const {
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);
  std::normal_distribution<float> perturbation(0.0f, options.intensity);

  for (auto& neuro : layerWeights) {
    for (auto& weight : neuro) {
      if (chance(random_engine) < options.rate) {
        weight += perturbation(random_engine);
      }
    }
  }
}

void GeneticTrainer::mutateBiases(layer_bias_t& layerBiases) const {
  std::uniform_real_distribution<float> chance(0.0f, 1.0f);
  std::normal_distribution<float> perturbation(0.0f, options.intensity);

  for (auto& bias : layerBiases) {
    if (chance(random_engine) < options.rate) {
      bias += perturbation(random_engine);
    }
  }
}

layer_weight_t GeneticTrainer::crossoverWeights(const layer_weight_t& partnerA, const layer_weight_t& partnerB) const {
  std::uniform_int_distribution<int> choose(0, 1);

  auto childWeights = partnerA;

  for (size_t i = 0; i < partnerA.size(); ++i) {
    for (size_t j = 0; j < partnerA[i].size(); ++j) {
      if (choose(random_engine) == 1) {
        childWeights[i][j] = partnerB[i][j];
      }
    }
  }

  return childWeights;
}

layer_bias_t GeneticTrainer::crossoverBiases(const layer_bias_t& partnerA, const layer_bias_t& partnerB) const {
  std::uniform_int_distribution<int> choose(0, 1);

  auto childBiases = partnerA;

  for (size_t i = 0; i < partnerA.size(); ++i) {
    if (choose(random_engine) == 1) {
      childBiases[i] = partnerB[i];
    }
  }

  return childBiases;
}

std::unique_ptr<IPopulation> GeneticTrainer::evolve(const IPopulation& population) const {
  std::vector<size_t> indices(population.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::partial_sort(
      indices.begin(), indices.begin() + options.eliteCount, indices.end(),
      [&](size_t a, size_t b) {
        return population[a].getFitness() > population[b].getFitness();
      });

  auto newPopulation = std::make_unique<Population>();

  newPopulation->reserve(population.size());

  for (size_t i = 0; i < options.eliteCount && i < indices.size(); ++i) {
    newPopulation->addIndividual(population[indices[i]]);
  }

  while (newPopulation->size() < population.size()) {
    auto* parentA = select(population);
    auto* parentB = select(population);

    if (!parentA || !parentB) {
      continue;
    }

    newPopulation->addIndividual(crossover(*parentA, *parentB));
  }

  return newPopulation;
}

std::vector<IIndividual> GeneticTrainer::evolve(const std::vector<IIndividual>& individuals) const {
  return std::vector<IIndividual>();
}

const IIndividual* GeneticTrainer::select(const IPopulation& population) const {
  if (population.empty()) {
    return nullptr;
  }

  std::srand(std::time(nullptr));

  int idx1 = std::rand() % population.size();
  int idx2 = std::rand() % population.size();

  auto& individualA = population[idx1];

  if (population[idx1].getFitness() > population[idx2].getFitness()) {
    return &population[idx1];
  }

  return &population[idx2];
}

const IIndividual* GeneticTrainer::select(const std::vector<IIndividual>& individuals) const {
  return nullptr;
}

void GeneticTrainer::setOptions(const GeneticOptions& options) {
  this->options = options;
}

void GeneticTrainer::setRate(float rate) {
  options.rate = rate;
}

void GeneticTrainer::setIntensity(float intensity) {
  options.intensity = intensity;
}

void GeneticTrainer::setEliteCount(size_t eliteCount) {
  options.eliteCount = eliteCount;
}

const GeneticOptions& GeneticTrainer::getOptions() const {
  return options;
}

};  // namespace neuro
