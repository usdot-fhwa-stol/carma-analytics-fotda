#!/bin/bash

apt-get update 

# Dependencies
dependencies="python3 \
    python3-pip
    python3-rosbag"

# Required python packages
python_packages="
    bagpy
    mpld3
    plotly
    lz4"

python3.8 -m pip install roslz4 --extra-index-url https://rospypi.github.io/simple/
pip install ipykernel
pip install --upgrade nbformat

# Install dependencies, packages
apt-get install -y $dependencies
pip3 install $python_packages
