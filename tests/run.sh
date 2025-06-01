#!/bin/bash

test_name=""

while getopts "n:" opt; do
  case $opt in
  n)
    test_name="$OPTARG"
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

echo "-- Start application"
echo ""

test_executable="./build/$test_name.exe"

$test_executable

echo ""
echo "-- Close application"

exit 0
