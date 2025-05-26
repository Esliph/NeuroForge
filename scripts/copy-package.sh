#!/bin/bash

echo "-- Copy file executable to packages folder"

mkdir -p packages && cp -r ./build/NeuroForge.exe ./packages/NeuroForge.exe

echo "-- Copy file executable completed"

exit 0
