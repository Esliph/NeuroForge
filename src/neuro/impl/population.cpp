#include "neuro/impl/population.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "neuro/exceptions/invalid_network_architecture_exception.hpp"
#include "neuro/impl/individual.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  Population::Population(const Population& population) {
    for (const auto& individual : population) {
      this->individuals.push_back(std::move(individual->clone()));
    }
  }

  Population::Population(const std::vector<IIndividual>& individuals) {
    for (const auto& individual : individuals) {
      this->individuals.push_back(std::move(individual.clone()));
    }
  }

  Population::Population(std::vector<std::shared_ptr<IIndividual>>& individuals)
      : individuals(std::move(individuals)) {}

  Population::Population(size_t size, const std::vector<int>& structure, const ActivationFunction& activation) {
    for (size_t i = 0; i < size; i++) {
      individuals.emplace_back(std::make_shared<Individual>(structure, activation));
    }
  }

  Population::Population(size_t size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations) {
    for (size_t i = 0; i < size; i++) {
      individuals.emplace_back(std::make_shared<Individual>(structure, activations));
    }
  }

  void Population::randomizeWeights(float min, float max) {
    for (const auto& individual : individuals) {
      individual->randomizeWeights(min, max);
    }
  }

  void Population::randomizeBiases(float min, float max) {
    for (const auto& individual : individuals) {
      individual->randomizeBiases(min, max);
    }
  }

  void Population::addIndividuals(const std::vector<IIndividual>& individuals) {
    for (const auto& individual : individuals) {
      this->individuals.push_back(std::move(individual.clone()));
    }
  }

  void Population::addIndividual(const IIndividual& individual) {
    individuals.push_back(individual.clone());
  }

  void Population::addIndividuals(std::vector<std::shared_ptr<IIndividual>>& individuals) {
    for (auto& individual : individuals) {
      this->individuals.push_back(individual);
    }
  }

  void Population::addIndividual(std::shared_ptr<IIndividual> individual) {
    individuals.push_back(individual);
  }

  void Population::removeIndividual(size_t index) {
    if (index < individuals.size()) {
      individuals.erase(individuals.begin() + index);
    }
  }

  void Population::clearIndividuals() {
    individuals.clear();
  }

  void Population::popIndividual() {
    individuals.pop_back();
  }

  void Population::reserve(size_t size) {
    individuals.reserve(size);
  }

  const IIndividual& Population::getBestIndividual() const {
    return **std::max_element(individuals.begin(), individuals.end(), [](const std::shared_ptr<IIndividual>& individualA, const std::shared_ptr<IIndividual>& individualB) {
      return individualA->getFitness() > individualB->getFitness();
    });
  }

  const std::vector<std::shared_ptr<IIndividual>>& Population::getIndividuals() const {
    return individuals;
  }

  std::vector<std::shared_ptr<IIndividual>>& Population::getIndividuals() {
    return individuals;
  }

  const IIndividual& Population::get(size_t index) const {
    if (individuals.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Individual vector out-of-range index");
    }

    return *individuals[index];
  }

  IIndividual& Population::get(size_t index) {
    if (individuals.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Individual vector out-of-range index");
    }

    return *individuals[index];
  }

  size_t Population::size() const {
    return individuals.size();
  }

  bool Population::empty() const {
    return individuals.size() == 0;
  }

  std::vector<std::shared_ptr<IIndividual>>::const_iterator Population::begin() const {
    return individuals.begin();
  }

  std::vector<std::shared_ptr<IIndividual>>::iterator Population::begin() {
    return individuals.begin();
  }

  std::vector<std::shared_ptr<IIndividual>>::const_iterator Population::end() const {
    return individuals.end();
  }

  std::vector<std::shared_ptr<IIndividual>>::iterator Population::end() {
    return individuals.end();
  }

  const IIndividual& Population::operator[](int index) const {
    if (individuals.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Individual vector out-of-range index");
    }

    return *individuals[index];
  }

  IIndividual& Population::operator[](int index) {
    if (individuals.size() >= index) {
      throw exception::InvalidNetworkArchitectureException("Individual vector out-of-range index");
    }

    return *individuals[index];
  }

  std::unique_ptr<IPopulation> Population::clone() const {
    return std::make_unique<Population>(*this);
  }
};  // namespace neuro
