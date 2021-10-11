#!/bin/bash

# Get the repo location. Assumes that this
# script is executed from inside the repo.
BASEDIR=`cd .. && pwd`
sudo docker run -it --runtime=nvidia -v ${BASEDIR}:/mount sputnik-dev
