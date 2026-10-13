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

ARG DOCKER_ORG="usdotfhwastoldev"
ARG DOCKER_TAG="develop-humble"
FROM ${DOCKER_ORG}/carma-base:${DOCKER_TAG}

LABEL Description="CDASim and CARMA Platform data analysis tools"

USER carma
WORKDIR /home/carma

# Install dependencies before copying analysis sources so source-only changes
# can reuse the expensive ROS message workspace layer.
COPY --chown=carma:carma docker/cdasim-data-analysis/install.sh ./docker/install.sh
RUN /home/carma/docker/install.sh

# The image is built from this repository checkout. Do not clone a moving
# carma-analytics-fotda branch during the image build.
COPY --chown=carma:carma src/cdasim_data_analysis_scripts ./cdasim_data_analysis_scripts
COPY --chown=carma:carma src/carma-platform ./carma-platform-scripts

RUN test -f /home/carma/cdasim_data_analysis_scripts/cdasim_log_analyzer.py && \
    test -f /home/carma/cdasim_data_analysis_scripts/cdasim_mcap_analyzer.py && \
    test -f /home/carma/carma-platform-scripts/run_all_control_analysis.py && \
    test -f /home/carma/carma-platform-scripts/run_all_regression_analysis.py

RUN echo "source /opt/ros/humble/setup.bash" >> /home/carma/.bashrc && \
    echo "source /home/carma/msgs_ws/install/setup.bash" >> /home/carma/.bashrc

ENV ROS_DISTRO=humble
SHELL ["/bin/bash", "-c"]
CMD ["bash"]
