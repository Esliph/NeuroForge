BUILD_DIR = build
CMAKE_GENERATOR = "MinGW Makefiles"
CLEAN ?= 0
RUN ?= 0

build:
	@if [ "$(CLEAN)" -eq 1 ]; then \
		echo "-- Cleaning build"; \
		rm -rf $(BUILD_DIR)/production; \
	fi

	@cmake -B $(BUILD_DIR)/production -G $(CMAKE_GENERATOR) -DENVIRONMENT=production
	@cmake --build $(BUILD_DIR)/production --target install
.PHONY: build

dev:
	@if [ "$(CLEAN)" -eq 1 ]; then \
		echo "-- Cleaning build"; \
		rm -rf $(BUILD_DIR)/dev; \
	fi

	@cmake -B $(BUILD_DIR)/dev -G $(CMAKE_GENERATOR) -DENVIRONMENT=development
	@cmake --build $(BUILD_DIR)/dev

	@if [ "$(RUN)" -eq 1 ]; then \
		echo "-- Running executable"; \
		./$(BUILD_DIR)/dev/NeuroForgeMain; \
	fi
.PHONY: dev

tests:
	@if [ "$(CLEAN)" -eq 1 ]; then \
		echo "-- Cleaning build"; \
		rm -rf $(BUILD_DIR)/tests; \
	fi

	@cmake -B $(BUILD_DIR)/tests -G $(CMAKE_GENERATOR) -DENVIRONMENT=tests
	@cmake --build $(BUILD_DIR)/tests

	@if [ "$(RUN)" -eq 1 ]; then \
		echo "-- Running tests"; \
		./$(BUILD_DIR)/tests/tests/NeuroForgeTests; \
	fi
.PHONY: tests

all: build
