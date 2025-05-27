#!/bin/bash

./scripts/build.sh "$@"

if [[ $? -ne 0 ]]; then
  exit 1
fi

./scripts/start.sh

exit 0
