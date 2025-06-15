#!/bin/bash

is_clean_build=false
test_name=""
is_run=false

while getopts "cn:r" opt; do
  case $opt in
  c)
    is_clean_build=true
    ;;
  n)
    test_name="$OPTARG"
    ;;
  r)
    is_run=true
    ;;
  \?)
    echo "Option invalid: -$OPTARG" >&2
    exit 1
    ;;
  :)
    echo "The option -$OPTARG required a argument." >&2
    exit 1
    ;;
  esac
done

if [ ! -d $test_name ]; then
  echo "Error: Test not found!"
  exit 1
fi

cd $test_name

if [[ $is_clean_build == true ]]; then
  echo "-- Cleaning test"

  rm -r ./build --force

  echo "-- Test cleaned"
fi

echo "-- Start building test"

cmake -B build -G "MinGW Makefiles" && cmake --build build

if [ $? -ne 0 ]; then
  echo "-- Build failed"
  exit 1
fi

echo "-- Build success"

cp -r "./build/$test_name.exe" "./$test_name.exe"

if [[ $is_run == true ]]; then
  cd ..

  ./run.sh -n $test_name
fi

exit 0
