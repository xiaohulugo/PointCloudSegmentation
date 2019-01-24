#! /bin/bash

# This script will build code and run the point cloud segmentation

mkdir -p /pointSegment/build && cd /pointSegment/build

cmake ..
make

./PointCloudSegmentation