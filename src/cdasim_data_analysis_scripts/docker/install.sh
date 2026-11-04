#!/bin/bash
#  Copyright (C) 2026 LEIDOS.
#
#  Licensed under the Apache License, Version 2.0 (the "License"); you may not
#  use this file except in compliance with the License. You may obtain a copy of
#  the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations under
#  the License.

set -e

echo "### Installing CDASim data analysis dependencies ###"

sudo apt-get update
sudo apt-get install -y --no-install-recommends --no-upgrade \
    git \
    python3-pip
sudo rm -rf /var/lib/apt/lists/*

python3 -m pip install --no-cache-dir \
    argcomplete \
    matplotlib \
    mcap-ros2-support \
    numpy \
    pytest \
    scipy

source /opt/ros/humble/setup.bash

mkdir -p /home/carma/msgs_ws/src
git clone --depth 1 --branch develop \
    https://github.com/usdot-fhwa-stol/carma-msgs.git \
    /home/carma/msgs_ws/src/carma-msgs

git clone --depth 1 --branch carma-develop \
    https://github.com/usdot-fhwa-stol/autoware.ai.git \
    /home/carma/autoware_msgs_src/autoware.ai
ln -s \
    /home/carma/autoware_msgs_src/autoware.ai/messages/autoware_msgs \
    /home/carma/msgs_ws/src/autoware_msgs
ln -s \
    /home/carma/autoware_msgs_src/autoware.ai/jsk_recognition/jsk_recognition_msgs \
    /home/carma/msgs_ws/src/jsk_recognition_msgs

cd /home/carma/msgs_ws
colcon build --symlink-install \
    --packages-up-to carma_planning_msgs autoware_msgs

echo "### CDASim data analysis dependencies installed ###"
