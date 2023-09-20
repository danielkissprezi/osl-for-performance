#!/usr/bin/env bash

set -ex

rm -f docs/combined-histogram.png
convert `find docs -iname "histogram*png" | sort -V` +append docs/combined-histogram.png;
