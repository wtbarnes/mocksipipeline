"""
Quick script to generate CHIANTI isothermal spectral table, separated  by element
"""
import pathlib

import astropy.units as u
import hissw
import numpy as np
from synthesizAR import log
from synthesizAR.atomic.idl import compute_spectral_table, write_spectral_table

log.setLevel('DEBUG')

SPEC_DIR = pathlib.Path(__file__).parent.parent.resolve() / 'mocksipipeline' / 'spectral' / 'data'


def list_all_ions(sort_by_element=True):
    """
    Get list of all ions in CHIANTI, sorted by element into a dictionary.
    """
    # This is just for sorting the list by ion
    def sort_func(x):
        el, ion_num = x.split('_')
        if 'd' in ion_num:
            ion_num = ion_num[:-1]
        return el, int(ion_num)

    env = hissw.Environment(ssw_packages=['chianti'],)
    res = env.run("read_masterlist, !xuvtop+'/masterlist/masterlist.ions', mlist", verbose=False)
    ion_list = [i.decode('utf-8') for i in res['mlist']]
    if not sort_by_element:
        return sorted(ion_list, key=sort_func)
    ion_dict = {}
    for i in ion_list:
        el, _ = i.split('_')
        if el not in ion_dict:
            ion_dict[el] = [i,]
        else:
            ion_dict[el].append(i)
    return {k: sorted(ion_dict[k], key=sort_func) for k in ion_dict}


if __name__ == '__main__':
    """
    This script computes a set of isothermal spectra for a variety of cases
    1. coronal abundances
    2. photospheric abundances
    3. unity abundances, separated by ion (effectively abundance free)
    """

    # NOTE: These are tuned to relevant ranges for MOXSI
    temperature = 10**np.arange(5, 8, 0.05) * u.K
    constant_pressure = 1e15 * u.K * u.cm**(-3)
    density = constant_pressure / temperature
    wave_min = 0.1 * u.Angstrom
    # wave_max = 500 * u.Angstrom
    wave_max = 100 * u.Angstrom
    # delta_wave = 71.8 * u.milliangstrom / 8
    delta_wave = 71.8 * u.milliangstrom / 2

    # Get list of all ions
    ion_catalogue = list_all_ions(sort_by_element=False)

    # Generate spectra for each element
    for k in ion_catalogue:
        spec_table = compute_spectral_table(
            temperature=temperature,
            density=density,
            wave_min=wave_min,
            wave_max=wave_max,
            delta_wave=delta_wave,
            ioneq_filename='chianti.ioneq',
            abundance_filename='unity.abund',  # independent of abundance
            include_continuum=False,
            use_lookup_table=True,
            include_all_lines=True,
            ion_list=[k],
        )
        filename = SPEC_DIR / f'chianti-spectrum-{k}.asdf'
        filename.parent.mkdir(parents=True, exist_ok=True)
        write_spectral_table(filename, spec_table)

    # Generate spectra for coronal abundances
    spec_table = compute_spectral_table(
        temperature=temperature,
        density=density,
        wave_min=wave_min,
        wave_max=wave_max,
        delta_wave=delta_wave,
        ioneq_filename='chianti.ioneq',
        abundance_filename='sun_coronal_1992_feldman_ext.abund',
        include_continuum=True,
        use_lookup_table=True,
        include_all_lines=True,
        ion_list=None,
    )
    filename = SPEC_DIR / 'chianti-spectrum-coronal-full.asdf'
    write_spectral_table(filename, spec_table)

    # Generate spectra for photospheric abundances
    spec_table = compute_spectral_table(
        temperature=temperature,
        density=density,
        wave_min=wave_min,
        wave_max=wave_max,
        delta_wave=delta_wave,
        ioneq_filename='chianti.ioneq',
        abundance_filename='sun_photospheric_2015_scott.abund',
        include_continuum=True,
        use_lookup_table=True,
        include_all_lines=True,
    )
    filename = SPEC_DIR / 'chianti-spectrum-photospheric.asdf'
    write_spectral_table(filename, spec_table)
