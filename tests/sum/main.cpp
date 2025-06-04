#define POPULATION_SIZE 200
#define GENERATIONS 150
#define MUTATION_RATE 0.1f
#define MUTATION_STRENGTH 0.4f
#define ELITE_COUNT 10

#include <iomanip>
#include <iostream>
#include <neuro/activation.hpp>
#include <neuro/individual.hpp>
#include <neuro/population.hpp>
#include <vector>

using namespace std;
using namespace neuro;

std::vector<std::pair<neuro_layer_t, float>> DATASET = {
    {{0.1f, 0.1f}, 0.2f},
    {{0.3f, 0.6f}, 0.9f},
    {{0.7f, 0.2f}, 0.9f},
    {{0.4f, 0.4f}, 0.8f},
    {{0.9f, 0.9f}, 1.8f},
    {{1.0f, 0.0f}, 1.0f},
};

float evaluateIndividual(const Individual& individual) {
  float totalError = 0.0f;
  const int testCases = 20;

  for (int i = 0; i < testCases; ++i) {
    float a = static_cast<float>(i) / testCases;
    float b = static_cast<float>(testCases - i) / testCases;

    neuro_layer_t input = {a, b};
    float expected = a + b;

    float output = individual.predict(input)[0];
    float error = expected - output;

    totalError += error * error;
  }

  return 1.0f / (1.0f + totalError);
}

int main() {
  vector<int> structure = {2, 6, 1};
  ActivationFunction activation = makeTanh_fn();

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
       << "Best individual (SUM):" << endl;

  for (const auto& [input, expected] : DATASET) {
    float output = best.getNeuralNetwork().feedforward(input)[0];

    cout << "Input: [" << input[0] << ", " << input[1] << "] | Expect: " << expected << " | Result: " << fixed << setprecision(4) << output << endl;
  }

  return 0;
}
