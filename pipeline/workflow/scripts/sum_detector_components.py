from overlappy.io import write_overlappogram

from mocksipipeline.util import stack_components

if __name__ == '__main__':
    # NOTE: Saving this with the WCS of the zeroth order dispersed image
    # Maybe there is a way to save the images with all of the WCS components
    # in additional HDUs?
    stacked_image = stack_components(snakemake.input, wcs_index=8)
    write_overlappogram(stacked_image, snakemake.output[0])
