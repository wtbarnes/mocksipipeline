import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame
from overlappy.io import write_overlappogram

import mocksipipeline.instrument.configuration
from mocksipipeline.modeling import (convolve_with_response,
                                     project_spectral_cube)
from mocksipipeline.util import read_data_cube

if __name__ == '__main__':
    # Read in config options
    dt = float(snakemake.config['exposure_time']) * u.s
    interval = float(snakemake.config['cadence']) * u.s
    apply_electron_conversion = bool(snakemake.config['apply_electron_conversion'])
    apply_gain_conversion = bool(snakemake.config['apply_gain_conversion'])
    include_psf = bool(snakemake.config['include_psf'])
    include_charge_spreading = bool(snakemake.config['include_charge_spreading'])
    if pointing_jitter := snakemake.config.get('pointing_jitter', None):
        pointing_jitter = float(pointing_jitter) * u.arcsec
    # Connect to scheduler if specified
    if client_address := snakemake.config.get('scheduler_address', None):
        import distributed
        client = distributed.Client(address=client_address)
    # Load instrument configuration
    instrument_design = getattr(mocksipipeline.instrument.configuration, snakemake.config['instrument_design'])
    # Select channel
    channel = instrument_design[snakemake.params.channel_name]
    # Read in spectral cube
    spec_cube = read_data_cube(snakemake.input[0])
    # Convolve with instrument response function
    instr_cube = convolve_with_response(spec_cube, channel, include_gain=False)
    # Remap spectral cube
    det_cube = project_spectral_cube(instr_cube,
                                     channel,
                                     wcs_to_celestial_frame(spec_cube.wcs).observer,
                                     dt=dt,
                                     interval=interval,
                                     pointing_jitter=pointing_jitter,
                                     include_psf=include_psf,
                                     include_charge_spreading=include_charge_spreading,
                                     apply_gain_conversion=apply_gain_conversion,
                                     apply_electron_conversion=apply_electron_conversion)
    # Save to file
    write_overlappogram(det_cube, snakemake.output[0])
