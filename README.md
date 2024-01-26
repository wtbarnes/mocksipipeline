# `mocksipipeline`


## Running the Pipeline

This pipeline uses the snakemake tool to generate synthetic MOXSI images.
Here is one example, where the images are being generated from AIA and XRT
observations from 2022-03-30 17:55 with coronal abundances

```shell
cd pipeline/workflow
$ snakemake ../results/flare-run/detector_images/all_components.fits \
            --config output_dir='flare-run' \
                     obstime='2022-03-30T17:55' \
                     instrument_design='moxsi_slot' \
            --cores 1
```

If you want to run the pipeline, but only rerun the steps if the needed files are missing (i.e. skip the checks on if any scripts have changed since the previous run),

```shell
$ snakemake ../results/flare-run/detector_images/all_components.fits
            --config output_dir='flare-run' \
                     obstime='2022-03-30T17:55'
                     instrument_design='moxsi_slot' \
            --cores 1 \
            --rerun-triggers mtime
```

If you want to produce a 1 s image in units of photons rather than DN,

```shell
$ snakemake ../results/ar-run/detector_images/all_components.fits \
            --config output_dir='ar-run' \
                     obstime='2020-11-09T17:59:57' \
                     exposure_time=1 \
                     cadence=1 \
                     instrument_design='moxsi_slot' \
                     convert_to_dn='False' \
            --cores 1
```
