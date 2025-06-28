BUILD_DIR = build
CMAKE_GENERATOR = "MinGW Makefiles"
CLEAN ?= 0

build:
	@if [ $(CLEAN) -eq 1 ]; then \
  	echo "-- Cleaning build"; \
		rm -rf $(BUILD_DIR)/production; \
	fi

	@cmake -B $(BUILD_DIR)/production -G $(CMAKE_GENERATOR) -DENVIRONMENT=production
	@cmake --build $(BUILD_DIR)/production --target install
.PHONY: build

tests:
	@if [ $(CLEAN) -eq 1 ]; then \
  	echo "-- Cleaning build"; \
		rm -rf $(BUILD_DIR)/tests; \
	fi

	@cmake -B $(BUILD_DIR)/tests -G $(CMAKE_GENERATOR) -DENVIRONMENT=tests
	@cmake --build $(BUILD_DIR)/tests

	@./$(BUILD_DIR)/tests/tests/NeuroForgeTests
.PHONY: tests

all: build_prod
