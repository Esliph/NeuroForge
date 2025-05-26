#!/bin/bash

is_clean_build=false
is_copy_exe=false

while getopts "cpa" opt; do
  case $opt in
  c)
    is_clean_build=true
    ;;
  p)
    is_copy_exe=true
    ;;
  a)
    is_clean_build=true
    is_copy_exe=true
    ;;
  esac
done

if [[ $is_clean_build == true ]]; then
  ./scripts/clean.sh
fi

echo "-- Start building"
mkdir -p build && cd build && cmake .. -G "MinGW Makefiles" .. && make

if [[ $? -ne 0 ]]; then
  echo "-- Build failed"
  exit 0
fi

echo "-- Build success"

if [[ $is_copy_exe == true ]]; then
  cd .. && ./scripts/copy-package.sh
fi

exit 0
