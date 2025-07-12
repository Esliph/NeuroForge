#pragma once

#include <memory>
#include <vector>

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
    Population(size_t size, const std::vector<int>& structure, const ActivationFunction& activation);
    Population(size_t size, const std::vector<int>& structure, const std::vector<ActivationFunction>& activations);

    virtual ~Population() = default;

    void randomizeWeights(float min, float max) override;
    void randomizeBiases(float min, float max) override;

    void addIndividuals(const std::vector<IIndividual>&) override;
    void addIndividual(const IIndividual&) override;
    void addIndividuals(std::vector<std::shared_ptr<IIndividual>>&) override;
    void addIndividual(std::shared_ptr<IIndividual>) override;

    void removeIndividual(size_t index) override;
    void clearIndividuals() override;
    void popIndividual() override;

    void reserve(size_t size) override;

    const IIndividual& getBestIndividual() const override;

    const std::vector<std::shared_ptr<IIndividual>>& getIndividuals() const override;
    std::vector<std::shared_ptr<IIndividual>>& getIndividuals() override;

    const IIndividual& get(size_t index) const override;
    IIndividual& get(size_t index) override;

    size_t size() const override;

    bool empty() const override;

    std::vector<std::shared_ptr<IIndividual>>::const_iterator begin() const override;
    std::vector<std::shared_ptr<IIndividual>>::iterator begin() override;

    std::vector<std::shared_ptr<IIndividual>>::const_iterator end() const override;
    std::vector<std::shared_ptr<IIndividual>>::iterator end() override;

    const IIndividual& operator[](size_t index) const override;
    IIndividual& operator[](size_t index) override;

    virtual std::unique_ptr<IPopulation> clone() const;
  };

};  // namespace neuro
