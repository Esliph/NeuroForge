#!/bin/bash

is_clean_build=false
is_copy_exe=false
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
  ./scripts/clean.sh
fi

echo "-- Start building"
mkdir -p build && cd build && cmake .. -G "MinGW Makefiles" && make

if [[ $? -ne 0 ]]; then
  echo "-- Build failed"
  exit 1
fi

echo "-- Build success"

if [[ $is_install == true ]]; then
  cd .. && ./scripts/make-install.sh
fi

exit 0
