#pragma once

#include <functional>
#include <memory>
#include <type_traits>

#include "neuro/core/i_neural_network.hpp"

namespace neuro {

  template <typename TINeuralNetwork = INeuralNetwork>
  class IIndividual {
    static_assert(std::is_base_of<INeuralNetwork, TINeuralNetwork>::value, "TINeuralNetwork must inherit from INeuralNetwork");

   public:
    IIndividual() = default;
    virtual ~IIndividual() = default;

    virtual void evaluateFitness(const std::function<float(const TINeuralNetwork&)>& evaluateFunction) = 0;

    virtual const TINeuralNetwork& getNeuralNetwork() const = 0;
    virtual TINeuralNetwork& getNeuralNetwork() = 0;

    virtual void setNeuralNetwork(const TINeuralNetwork&) = 0;
    virtual void setNeuralNetwork(std::unique_ptr<TINeuralNetwork>) = 0;

    virtual float getFitness() const = 0;
    virtual void setFitness(float) = 0;

    virtual std::unique_ptr<IIndividual> clone() const = 0;
  };

} // namespace neuro
