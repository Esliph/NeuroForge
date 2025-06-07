#define POPULATION_SIZE 100
#define GENERATIONS 100
#define MUTATION_RATE 0.2f
#define MUTATION_STRENGTH 0.8f
#define ELITE_COUNT 5

#include <iomanip>
#include <iostream>
#include <neuro/activation.hpp>
#include <neuro/individual.hpp>
#include <neuro/population.hpp>
#include <vector>

using namespace std;
using namespace neuro;

const vector<pair<neuro_layer_t, float>> DATASET = {
    {{0.0f, 0.0f}, 0.0f},
    {{0.0f, 1.0f}, 1.0f},
    {{1.0f, 0.0f}, 1.0f},
    {{1.0f, 1.0f}, 0.0f},
};

float evaluateIndividual(const Individual& individual) {
  float totalError = 0.0f;

  for (const auto& [input, expected] : DATASET) {
    float output = individual.predict(input)[0];
    float error = expected - output;

    totalError += error * error;
  }

  return 1.0f / (1.0f + totalError);
}

int main() {
  vector<int> structure = {2, 4, 1};
  ActivationFunction activation = makeSigmoid();

  Population population(POPULATION_SIZE, structure, activation);

  population.loadWeights(-5, 5);
  population.loadBias(-5, 5);

  for (int generation = 0; generation <= GENERATIONS; ++generation) {
    for (auto& individual : population.getIndividuals()) {
      float fitness = evaluateIndividual(individual);

      individual.setFitness(fitness);
    }

    const Individual& best = population.getBest();
    float bestFitness = best.getFitness();

    cout << "Generation " << setw(3) << generation << " | Best fitness: " << fixed << setprecision(6) << bestFitness << endl;

    population.evolve(MUTATION_RATE, MUTATION_STRENGTH, ELITE_COUNT);
  }

  const Individual& best = population.getBest();

  cout << endl
       << "Best individual (XOR):" << endl;

  for (const auto& [input, expected] : DATASET) {
    float output = best.getNeuralNetwork().feedforward(input)[0];

    cout << "Input: [" << input[0] << ", " << input[1] << "] | Expect: " << expected << " | Result: " << fixed << setprecision(4) << output << endl;
  }

  return 0;
}
