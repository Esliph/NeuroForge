#!/bin/bash

mkdir -p build && cd build && cmake -DENABLE_ASAN=ON .. && make
cd .. && valgrind ./scripts/start.sh
