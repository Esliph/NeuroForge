#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace neuro {

  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;
  using Duration = std::chrono::duration<double, std::micro>;

  class Measure {
    std::string name;

    std::vector<Clock::duration> durations;
    TimePoint start_time;

    bool running = false;

   public:
    Measure(const std::string& name = "")
        : name(name) {}

    void start() {
      if (running) {
        return;
      }

      start_time = Clock::now();

      running = true;
    }

    void stop() {
      if (!running) {
        return;
      }

      auto end_time = Clock::now();
      durations.push_back(end_time - start_time);

      running = false;
    }

    void reset() {
      durations.clear();
      running = false;
    }

    double last() const {
      if (durations.empty()) {
        return 0.0;
      }

      return Duration(durations.back()).count();
    }

    double average() const {
      if (durations.empty()) {
        return 0.0;
      }

      double sum = std::accumulate(durations.begin(), durations.end(), 0.0,
                                   [](double acc, const auto& d) { return acc + Duration(d).count(); });

      return sum / durations.size();
    }

    double min() const {
      if (durations.empty()) {
        return 0.0;
      }

      return Duration(*std::min_element(durations.begin(), durations.end())).count();
    }

    double max() const {
      if (durations.empty()) {
        return 0.0;
      }

      return Duration(*std::max_element(durations.begin(), durations.end())).count();
    }

    size_t count() const {
      return durations.size();
    }

    void print() const {
      std::cout << "[Measure: " << (name.empty() ? "Unnamed" : name) << "] "
                << "Count: " << count()
                << " | Avg: " << average() << " us"
                << " | Min: " << min() << " us"
                << " | Max: " << max() << " us"
                << " | Last: " << last() << " us\n";
    }

    static Measure run(const std::string& name, const std::function<void()>& handler, int iterations = 1, int warmup = 0) {
      Measure measure(name);

      for (int i = 0; i < warmup; i++) {
        handler();
      }

      for (int i = 0; i < iterations; i++) {
        measure.start();
        handler();
        measure.stop();
      }

      return measure;
    }
  };

};  // namespace neuro
