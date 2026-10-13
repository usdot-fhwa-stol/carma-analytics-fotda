# CDASim data analysis image

This image packages the CDASim analyzers in
`src/cdasim_data_analysis_scripts` and the CARMA Platform analyzers in
`src/carma-platform`. Both source directories are copied from the same
repository checkout used to build the image.

Build it locally from the repository root with:

```bash
./docker/cdasim-data-analysis/build-image.sh
```

Override the local image name or tag when needed:

```bash
IMAGE_NAME=usdotfhwastoldev/cdasim-data-analysis \
IMAGE_TAG=develop \
./docker/cdasim-data-analysis/build-image.sh
```
