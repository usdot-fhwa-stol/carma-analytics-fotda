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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DOCKER_ORG="${DOCKER_ORG:-usdotfhwastoldev}"
IMAGE_NAME="${IMAGE_NAME:-cdasim-data-analysis}"
IMAGE_TAG="${IMAGE_TAG:-develop}"
BASE_IMAGE_ORG="${BASE_IMAGE_ORG:-usdotfhwastoldev}"
BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-develop-humble}"
FULL_IMAGE_NAME="${DOCKER_ORG}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${FULL_IMAGE_NAME}"

docker build \
    --file "$SCRIPT_DIR/Dockerfile" \
    --tag "$FULL_IMAGE_NAME" \
    --build-arg "DOCKER_ORG=${BASE_IMAGE_ORG}" \
    --build-arg "DOCKER_TAG=${BASE_IMAGE_TAG}" \
    --build-arg "VERSION=${IMAGE_TAG}" \
    --build-arg "VCS_REF=$(git -C "$REPOSITORY_ROOT" rev-parse --short HEAD)" \
    --build-arg "BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    "$REPOSITORY_ROOT"

echo "Built ${FULL_IMAGE_NAME}"
