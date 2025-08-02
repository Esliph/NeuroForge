ifeq ($(OS),Windows_NT)
		ifeq ($(findstring Microsoft,$(shell uname -r)),Microsoft)
				CMAKE_GENERATOR := "Unix Makefiles"
		else
				CMAKE_GENERATOR := "MinGW Makefiles"
		endif
else
		CMAKE_GENERATOR := "Unix Makefiles"
endif

BUILD_DIR = build
TEST_TYPE ?= unit
TEST_NAME ?= ""

ifeq ($(TEST_TYPE),unit)
	TEST_NAME = NeuroForgeTests
endif

build:
	@cmake -B $(BUILD_DIR)/production -G $(CMAKE_GENERATOR) -DENVIRONMENT=production
	@cmake --build $(BUILD_DIR)/production --target install
.PHONY: build

dev:
	@cmake -B $(BUILD_DIR)/dev -G $(CMAKE_GENERATOR) -DENVIRONMENT=development
	@cmake --build $(BUILD_DIR)/dev
.PHONY: dev

tests:
	@cmake -S . -B $(BUILD_DIR)/tests -G $(CMAKE_GENERATOR) -DENVIRONMENT=tests -DTEST_TYPE=$(TEST_TYPE) -DTEST_NAME=$(TEST_NAME)
	@cmake --build $(BUILD_DIR)/tests
.PHONY: tests

run_dev:
	@make dev

	@echo "-- Running executable";
	@./$(BUILD_DIR)/dev/NeuroForgeMain;
.PHONY: run_build

run_tests:
	@make tests

	@echo "-- Running tests";
	@./$(BUILD_DIR)/tests/tests/$(TEST_NAME);
.PHONY: run_tests

all: build
