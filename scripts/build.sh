#!/bin/bash

env="PROD" # PROD | TEST | NONE
is_clean_build=false

exit_with_error() {
  echo "$1"
  exit 1
}

validate_env() {
  if [[ "$env" != "PROD" && "$env" != "TEST" && "$env" != "NONE" ]]; then
    exit_with_error "Invalid argument \"$env\" for flag \"-e\". Valid values are 'PROD', 'TEST' or 'NONE'."
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
  echo "-- Building Production"
  cmake -B build/production -G "MinGW Makefiles" -DCONSTRUCT_INSTALLATION=ON
  cmake --build build/production --target install
}

build_test() {
  echo "-- Building Tests"
  cmake -B build/tests -G "MinGW Makefiles" -DENABLE_TEST=ON
  cmake --build build/tests

  if [[ $? -ne 0 ]]; then
    exit_with_error "-- Build failed"
  fi
}

build_no_target() {
  echo "-- Building with no Target"
  cmake -B build/temp -G "MinGW Makefiles"
  cmake --build build/temp
}

start_build() {
  case "$env" in
  "PROD")
    build_production
    ;;
  "TEST")
    build_test
    ;;
  "NONE")
    build_no_target
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
