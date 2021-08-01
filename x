#!/bin/sh
set -e

./configure --examples

make example-c
./bin/example-c
