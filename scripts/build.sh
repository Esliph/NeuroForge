#!/bin/bash

env="PROD" # PROD | TEST
is_clean_build=false

exit_with_error() {
  echo "$1"
  exit 1
}

validate_env() {
  if [[ "$env" != "PROD" && "$env" != "TEST" ]]; then
    exit_with_error "Invalid argument \"$env\" for flag \"-e\". Valid values are 'PROD' or 'TEST'."
  fi
}

clean_project() {
  echo "-- Cleaning build"
  case "$env" in
  "PROD")
    rm -rf build/production
    ;;
  "TEST")
    rm -rf build/tests
    ;;
  "NONE")
    rm -rf build/temp
    ;;
  esac
  echo "-- Build cleaned"
}

build_production() {
  cmake -B build/production -G "MinGW Makefiles" -DENVIRONMENT=production
  cmake --build build/production --target install
}

build_test() {
  cmake -B build/tests -G "MinGW Makefiles" -DENVIRONMENT=tests
  cmake --build build/tests

  if [[ $? -ne 0 ]]; then
    exit_with_error "-- Build failed"
  fi

  ./build/tests/tests/NeuroForgeTests
}

start_build() {
  case "$env" in
  "PROD")
    build_production
    ;;
  "TEST")
    build_test
    ;;
  esac
}

while getopts "e:c" opt; do
  case $opt in
  c)
    is_clean_build=true
    ;;
  e)
    env="${OPTARG^^}"
    ;;
  *)
    exit_with_error "Invalid flag"
    ;;
  esac
done

validate_env

if $is_clean_build; then
  clean_project
fi

echo "-- Start building"

start_build

if [[ $? -ne 0 ]]; then
  exit_with_error "-- Build failed"
fi

echo "-- Build success"
exit 0
