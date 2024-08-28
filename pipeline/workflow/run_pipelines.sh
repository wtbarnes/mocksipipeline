#!/bin/zsh
# A place to put multiple pipeline commands if you need to run one after another

for element in fe si mg ne o
do
    # Coronal
    snakemake ../results/cdr/rfa/ar_1h_coronal_${element}/detector_images/all_components.fits \
              --config root_dir=../results/cdr/rfa \
                       output_dir=ar_1h_coronal_${element} \
                       instrument_design=moxsi_cdr \
                       spectral_table=../../mocksipipeline/spectral/data/chianti-spectrum-${element}-sun_coronal_1992_feldman_ext.asdf \
                       max_spectral_order=3 \
              --cores 1 \
              --rerun-triggers mtime
    # Photospheric
    snakemake ../results/cdr/rfa/ar_1h_photospheric_${element}/detector_images/all_components.fits \
              --config root_dir=../results/cdr/rfa \
                       output_dir=ar_1h_photospheric_${element} \
                       instrument_design=moxsi_cdr \
                       spectral_table=../../mocksipipeline/spectral/data/chianti-spectrum-${element}-sun_photospheric_2015_scott.asdf \
                       max_spectral_order=3 \
              --cores 1 \
              --rerun-triggers mtime
done
