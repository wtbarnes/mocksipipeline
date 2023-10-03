"""
Module for converting DEM to spectral cube
"""
import warnings

import ndcube
from astropy.utils.data import get_pkg_data_filenames

__all__ = ['calculate_intensity', 'SpectralModel', 'get_spectral_tables']


def calculate_intensity(em, spectral_table, header):
    from synthesizAR.instruments import InstrumentDEM
    return InstrumentDEM.calculate_intensity(em, spectral_table, header)


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
        return calculate_intensity(dem_cube, self.spectral_table, dict(celestial_wcs.to_header()))


def get_spectral_tables(pattern='', sum_tables=False):
    """
    Get CHIANTI spectra as a function of temperature and wavelength

    Returns either component spectra, based on selected pattern or
    a summation of all selected tables. Summing is useful, e.g. if 
    you want the spectra for all ions of a given element.

    Parameters
    -----------
    pattern: `str, optional
        The pattern to use when globbing the available spectral tables.
    sum_tables: `bool`, optional
        If True, sum all tables into a single table and return that single
        table. If False, a dictionary of all tables matching the glob
        pattern specified by ``pattern`` will be returned. You should really
        only use this if you're using a specific pattern that targets specific
        elements/ions.

    Returns
    -------
    : `dict` if ``sum_tables=False`` or `ndcube.NDCube` if ``sum_tables=True``
    """
    from synthesizAR.atomic.idl import read_spectral_table
    spectral_tables = {}
    filenames = get_pkg_data_filenames(
        'data',
        package='mocksipipeline.physics.spectral',
        pattern=f'chianti-spectrum-{pattern}*.asdf'
    )
    for fname in filenames:
        tab = read_spectral_table(fname)
        abund_name = tab.meta['abundance_filename'].split('.')[0]
        ion_list = tab.meta['ion_list']
        el = '-'.join(ion_list) if isinstance(ion_list, list) else ion_list
        spectral_tables[f"{abund_name}_{el}"] = tab

    if sum_tables:
        spectral_tables = [v for _, v in spectral_tables.items()]
        summed_spectral_table = spectral_tables[0]
        for tab in spectral_tables[1:]:
            if tab.meta['abundance_filename'] != summed_spectral_table.meta['abundance_filename']:
                warnings.warn('Adding spectral tables with different abundance filenames.')
            summed_spectral_table += tab.data*tab.unit
            summed_spectral_table.meta['ion_list'] += tab.meta['ion_list']
        return summed_spectral_table
    else:
        return spectral_tables
