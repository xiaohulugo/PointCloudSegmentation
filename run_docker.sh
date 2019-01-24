#! /bin/bash

# This script is used to run the docker container

docker run -it --rm \
    -v $(pwd):/pointSegment \
    opencv:cpp \
    /pointSegment/test_segment.sh