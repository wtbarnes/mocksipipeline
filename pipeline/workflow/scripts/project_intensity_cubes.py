import astropy.units as u
from overlappy.io import write_overlappogram

from mocksipipeline.detector import project_spectral_cube
from mocksipipeline.detector.response import Channel, SpectrogramChannel
from mocksipipeline.util import read_data_cube

if __name__ == '__main__':
    # Read in config options
    dt = float(snakemake.config['exposure_time']) * u.s
    interval = float(snakemake.config['cadence']) * u.s
    convert_to_dn = bool(snakemake.config['convert_to_dn'])
    # Create channel object
    channel_name = snakemake.params.channel
    spectral_order = int(snakemake.params.order)
    if 'filtergram' in channel_name:
        channel = Channel(channel_name)
    elif 'spectrogram' in channel_name:
        channel = SpectrogramChannel(spectral_order)
    else:
        raise ValueError(f'Unrecognized channel name {channel_name}')
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
