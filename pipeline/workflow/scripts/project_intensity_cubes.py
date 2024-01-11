import astropy.units as u
from overlappy.io import write_overlappogram

from mocksipipeline.detector import project_spectral_cube
from mocksipipeline.instrument.optics.response import Channel
from mocksipipeline.util import read_data_cube

if __name__ == '__main__':
    # Read in config options
    dt = float(snakemake.config['exposure_time']) * u.s
    interval = float(snakemake.config['cadence']) * u.s
    convert_to_dn = bool(snakemake.config['convert_to_dn'])
    # Create channel object
    channel_name = snakemake.params.channel
    spectral_order = int(snakemake.params.order)
    # TODO: add ability to specify different designs
    channel = Channel(channel_name, order=spectral_order)
    # Read in spectral cube
    spec_cube = read_data_cube(snakemake.input[0])
    # Remap spectral cube
    det_cube = project_spectral_cube(spec_cube,
                                     channel,
                                     dt=dt,
                                     interval=interval,
                                     convert_to_dn=convert_to_dn)
    # Save to file
    write_overlappogram(det_cube, snakemake.output[0])
