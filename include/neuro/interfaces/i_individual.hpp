#pragma once

#include <functional>
#include <memory>

#include "neuro/interfaces/i_layer.hpp"
#include "neuro/interfaces/i_neural_network.hpp"

namespace neuro {

  class IIndividual {
   public:
    IIndividual() = default;
    virtual ~IIndividual() = default;

    virtual void evaluateFitness(const std::function<float(const INeuralNetwork&)>& evaluateFunction) = 0;

    virtual void setNeuralNetwork(const INeuralNetwork&) = 0;
    virtual void setNeuralNetwork(std::unique_ptr<INeuralNetwork>) = 0;

    virtual void setFitness(float) = 0;

    virtual const INeuralNetwork& getNeuralNetwork() const = 0;
    virtual INeuralNetwork& getNeuralNetwork() = 0;

    virtual float getFitness() const = 0;

    virtual std::unique_ptr<IIndividual> clone() const = 0;
  };

};  // namespace neuro
