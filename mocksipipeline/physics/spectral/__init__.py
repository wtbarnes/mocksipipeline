"""
Module for converting DEM to spectral cube
"""
import ndcube
import numpy as np
import astropy.units as u
from astropy.utils.data import get_pkg_data_filenames

__all__ = ['SpectralModel', 'get_spectral_tables']


class SpectralModel:

    def __init__(self, spectral_table='sun_coronal_1992_feldman_ext_all', **kwargs):
        self.spectral_table = spectral_table

    @property
    def spectral_table(self):
        return self._spectral_table

    @spectral_table.setter
    def spectral_table(self, value):
        if isinstance(value, ndcube.NDCube):
            self._spectral_table = value
        else:
            spec_tables = get_spectral_tables()
            self._spectral_table = spec_tables[value]

    def run(self, dem_cube, celestial_wcs,):
        # TODO: figure out how to get the celestial WCS from the DEM cube, even if our dem cube has a gwcs
        # We can stop passing this in separately if we can figure out a sensible way
        # to get this out of our DEM cube. This is not currently possible due to the
        # fact that our DEM WCS is a gwcs.
        from synthesizAR.instruments import InstrumentDEM
        return InstrumentDEM.calculate_intensity(
            dem_cube,
            self.spectral_table,
            dict(celestial_wcs.to_header())
        )


def get_spectral_tables():
    from synthesizAR.atomic.idl import read_spectral_table
    spectral_tables = {}
    filenames = get_pkg_data_filenames(
        'data',
        package='mocksipipeline.physics.spectral',
        pattern='*.asdf'
    )
    for fname in filenames:
        tab = read_spectral_table(fname)
        abund_name = tab.meta['abundance_filename'].split('.')[0]
        ion_list = tab.meta['ion_list']
        if isinstance(ion_list, list):
            # NOTE: this assumes all ions in the list are the same element
            # which in general may not be true
            el = ion_list[0].split("_")[0]
        else:
            el = ion_list
        spectral_tables[f"{abund_name}_{el}"] = tab

    return spectral_tables
