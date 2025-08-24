#pragma once

#include <memory>
#include <vector>

#include "internal/attribute.hpp"
#include "neuro/interfaces/i_individual.hpp"
#include "neuro/interfaces/i_population.hpp"
#include "neuro/utils/activation.hpp"

namespace neuro {

  class Population : public IPopulation {
    std::vector<std::shared_ptr<IIndividual>> individuals{};

   public:
    Population() = default;
    Population(const Population&);

    Population(const std::vector<IIndividual>& individuals);
    Population(std::vector<std::shared_ptr<IIndividual>>& individuals);
    Population(size_t size, const std::vector<int>& structure);
    Population(size_t size, const std::vector<int>& structure, const ActivationFunction& activation);
    Population(size_t size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    virtual ~Population() = default;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    void addIndividuals(const std::vector<IIndividual>&) override;

    FORCE_INLINE void addIndividual(const IIndividual& individual) {
      individuals.push_back(individual.clone());
    }

    void addIndividuals(std::vector<std::shared_ptr<IIndividual>>&) override;

    FORCE_INLINE void addIndividual(std::shared_ptr<IIndividual> individual) {
      individuals.push_back(individual);
    }

    void removeIndividual(size_t index) override;

    FORCE_INLINE void clearIndividuals() {
      individuals.clear();
    }

    FORCE_INLINE void popIndividual() {
      individuals.pop_back();
    }

    FORCE_INLINE void reserve(size_t size) {
      individuals.reserve(size);
    }

    const IIndividual& getBestIndividual() const override;

    FORCE_INLINE const std::vector<std::shared_ptr<IIndividual>>& getIndividuals() const {
      return individuals;
    }

    FORCE_INLINE std::vector<std::shared_ptr<IIndividual>>& getIndividuals() {
      return individuals;
    }

    const IIndividual& get(size_t index) const override;
    IIndividual& get(size_t index) override;

    FORCE_INLINE size_t size() const {
      return individuals.size();
    }

    FORCE_INLINE bool empty() const {
      return individuals.empty();
    }

    FORCE_INLINE std::vector<std::shared_ptr<IIndividual>>::const_iterator begin() const {
      return individuals.begin();
    }

    FORCE_INLINE std::vector<std::shared_ptr<IIndividual>>::iterator begin() {
      return individuals.begin();
    }

    FORCE_INLINE std::vector<std::shared_ptr<IIndividual>>::const_iterator end() const {
      return individuals.end();
    }

    FORCE_INLINE std::vector<std::shared_ptr<IIndividual>>::iterator end() {
      return individuals.end();
    }

    const IIndividual& operator[](size_t index) const override;
    IIndividual& operator[](size_t index) override;

    FORCE_INLINE std::unique_ptr<IPopulation> clone() const {
      return std::make_unique<Population>(*this);
    }
  };

}; // namespace neuro
