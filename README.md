# `mocksipipeline`


## Running the Pipeline

This pipeline uses the snakemake tool to generate synthetic MOXSI images.
Here is one example, where the images are being generated from AIA and XRT
observations from 2022-03-30 17:55 with coronal abundances

```shell
cd pipeline/workflow
$ snakemake ../results/2022-03-30T17:55_sun_coronal_1992_feldman_ext_all/detector_images/all_components.fits --config root_dir='../results' obstime='2022-03-30T17:55' --cores 1
```

If you want to run the pipeline, but only rerun the steps if the needed files are missing (i.e. skip the checks on if any scripts have changed since the previous run),

```shell
$ snakemake ../results/2022-03-30T17:55_sun_coronal_1992_feldman_ext_all/detector_images/all_components.fits --config root_dir='../results' obstime='2022-03-30T17:55' --cores 1 --rerun-triggers mtime
```
