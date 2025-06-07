#!/bin/bash

is_clean_build=false
is_install=false

while getopts "ci" opt; do
  case $opt in
  c)
    is_clean_build=true
    ;;
  i)
    is_install=true
    ;;
  esac
done

if [[ $is_clean_build == true ]]; then
  echo "-- Cleaning project"

  rm -r build --force

  echo "-- Project cleaned"
fi

echo "-- Start building"

cmake -B build -G "MinGW Makefiles"

if [[ $is_install == true ]]; then
  cmake --build build --target install
else
  cmake --build build
fi

if [[ $? -ne 0 ]]; then
  echo "-- Build failed"
  exit 1
fi

echo "-- Build success"

exit 0
