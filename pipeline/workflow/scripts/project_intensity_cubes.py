import astropy.units as u
import scipy.ndimage
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import wcs_to_celestial_frame
from overlappy.io import write_overlappogram

import mocksipipeline.instrument.configuration
from mocksipipeline.modeling import (convolve_with_response,
                                     project_spectral_cube)
from mocksipipeline.util import read_data_cube


def apply_psf(cube, channel):
    # FIXME: This is just a placeholder PSF calculation. Need to implement an actual
    # convolution with a model PSF that more accurately captures the effect of the slot
    # on the spatial resolution
    # NOTE: By applying the PSF on the spectral cube, we are assuming that the pixel grid
    # here is aligned with the pixel grid of the detector. The PSF FWHM has been carefully
    # constructed to account for this. In general, we should figure out a way to define
    # the PSF kernel on the zeroth order detector grid and transform it to this coordinate
    # system.
    psf_sigma = channel.aperture.psf_fwhm * gaussian_fwhm_to_sigma / channel.spatial_plate_scale
    _ = scipy.ndimage.gaussian_filter(cube.data,
                                      psf_sigma.to_value('pixel')[::-1],
                                      axes=(1,2),
                                      output=cube.data)
    return cube


if __name__ == '__main__':
    # Read in config options
    dt = float(snakemake.config['exposure_time']) * u.s
    interval = float(snakemake.config['cadence']) * u.s
    apply_electron_conversion = bool(snakemake.config['apply_electron_conversion'])
    apply_gain_conversion = bool(snakemake.config['apply_gain_conversion'])
    # Load instrument configuration
    instrument_design = getattr(mocksipipeline.instrument.configuration, snakemake.config['instrument_design'])
    # Select channel
    channel = instrument_design[snakemake.params.channel_name]
    # Read in spectral cube
    spec_cube = read_data_cube(snakemake.input[0])
    # Convolve with instrument response function
    instr_cube = convolve_with_response(spec_cube, channel, include_gain=False)
    # Apply channel-dependent PSF
    # PSF convolution calculation goes here
    instr_cube = apply_psf(instr_cube, channel)
    # Remap spectral cube
    det_cube = project_spectral_cube(instr_cube,
                                     channel,
                                     wcs_to_celestial_frame(spec_cube.wcs).observer,
                                     dt=dt,
                                     interval=interval,
                                     apply_gain_conversion=apply_gain_conversion,
                                     apply_electron_conversion=apply_electron_conversion)
    # Save to file
    write_overlappogram(det_cube, snakemake.output[0])
