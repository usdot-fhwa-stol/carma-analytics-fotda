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
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cdasim-data-analysis}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

docker build \
    --file "$SCRIPT_DIR/Dockerfile" \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    "$REPOSITORY_ROOT"

echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
