#!/bin/bash

is_clean_build=false

while getopts "c" opt; do
  case $opt in
  c)
    is_clean_build=true
    ;;
  esac
done

if [[ $is_clean_build == true ]]; then
  echo "-- Cleaning project"

  rm -r build --force

  echo "-- Project cleaned"
fi

echo "-- Start building"

cmake -B build -G "MinGW Makefiles" && cmake --build build --target install

if [[ $? -ne 0 ]]; then
  echo "-- Build failed"
  exit 1
fi

echo "-- Build success"

exit 0
