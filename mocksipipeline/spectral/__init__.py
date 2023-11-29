"""
Module for converting DEM to spectral cube
"""
import warnings

import astropy.units as u
from astropy.utils.data import get_pkg_data_filenames
import numpy as np
from scipy.interpolate import interp1d

__all__ = ['calculate_intensity', 'get_spectral_tables', 'compute_temperature_response']


def calculate_intensity(em, spectral_table, header):
    from synthesizAR.instruments import InstrumentDEM
    return InstrumentDEM.calculate_intensity(em, spectral_table, header)


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


def compute_temperature_response(spectra,
                                 instrument_wavelength,
                                 instrument_response,
                                 return_temperature=False):
    """
    Generate a temperature response from a spectra and a wavelength response.

    Parameters
    ----------
    spectra: `~ndcube.NDCube`
        Isothermal spectra as a function of temperature and wavelength.
    instrument_wavelength: `~astropy.units.Quantity`
    instrument_response: `~astropy.units.Quantity`
    """
    # Interpolate response to spectra
    spectra_wavelength = spectra.axis_world_coords(1)[0].to('Angstrom')
    f_response = interp1d(instrument_wavelength.to_value(spectra_wavelength.unit),
                          instrument_response.to_value(),
                          axis=0,
                          bounds_error=False,
                          fill_value=0.0,)
    spectra_response = u.Quantity(f_response(spectra_wavelength.value), instrument_response.unit)
    # Integrate over wavelength
    delta_wave = np.diff(spectra_wavelength)[0]  # It is assumed that this is uniform
    spectra_data = u.Quantity(spectra.data, spectra.unit)
    temperature_response = (spectra_data * spectra_response * delta_wave).sum(axis=1)
    if return_temperature:
        temperature = spectra.axis_world_coords(0)[0].to('K')
        return temperature, temperature_response
    else:
        return temperature_response
