BUILD_DIR = build
CMAKE_GENERATOR = "MinGW Makefiles"
CLEAN ?= 1
RUN ?= 1
TEST_TYPE ?= "unit"
TEST_NAME ?= ""

ifeq ($(TEST_TYPE),"unit")
	TEST_NAME = "NeuroForgeMain"
endif

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

	@cmake -B $(BUILD_DIR)/tests -G $(CMAKE_GENERATOR) -DENVIRONMENT=tests -DTEST_TYPE=$(TEST_TYPE) -DTEST_NAME=$(TEST_NAME)
	@cmake --build $(BUILD_DIR)/tests

	@if [ "$(RUN)" -eq 1 ]; then \
		echo "-- Running tests"; \
		./$(BUILD_DIR)/tests/tests/$(TEST_NAME); \
	fi
.PHONY: tests

all: build
