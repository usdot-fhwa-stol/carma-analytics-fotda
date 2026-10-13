# CDASim data analysis image

This image packages the CDASim analyzers in
`src/cdasim_data_analysis_scripts` and the CARMA Platform analyzers in
`src/carma-platform`. Both source directories are copied from the same
repository checkout used to build the image.

Build it locally from the repository root with:

```bash
./src/cdasim_data_analysis_scripts/docker/build-image.sh
```

The default local image is:

```text
usdotfhwastoldev/cdasim-data-analysis:develop
```

Override the organization, image name, tag, or base image when needed:

```bash
DOCKER_ORG=usdotfhwastol \
IMAGE_NAME=cdasim-data-analysis \
IMAGE_TAG=carma-system-x.y.z \
BASE_IMAGE_ORG=usdotfhwastol \
BASE_IMAGE_TAG=carma-system-x.y.z-humble \
./src/cdasim_data_analysis_scripts/docker/build-image.sh
```
