import pathlib

import sunpy.io._fits as sunpy_fits

from mocksipipeline.spectral import calculate_intensity, get_spectral_tables
from mocksipipeline.util import read_cube_with_xarray

if __name__ == '__main__':
    em_cube = read_cube_with_xarray(snakemake.input[0], 'temperature', 'phys.temperature')
    spectral_table_name = snakemake.config['spectral_table']
    if pathlib.Path(spectral_table_name).is_file():
        from synthesizAR.atomic.idl import read_spectral_table
        spectral_table = read_spectral_table(spectral_table_name)
    else:
        spectral_table = get_spectral_tables()[spectral_table_name]
    # NOTE: the FITS WCS info is stored in the NDCube metadata
    spectral_cube = calculate_intensity(em_cube, spectral_table, em_cube.meta)
    sunpy_fits.write(snakemake.output[0],
                     spectral_cube.data,
                     spectral_cube.meta,
                     overwrite=True)
