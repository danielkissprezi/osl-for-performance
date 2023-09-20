#!/usr/bin/env bash

set -ex

rm -rf docs
mkdir docs
./main.py
./concat_histo.sh
