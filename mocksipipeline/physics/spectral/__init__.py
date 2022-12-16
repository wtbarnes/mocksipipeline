"""
Module for converting DEM to spectral cube
"""
import ndcube
import numpy as np
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename

__all__ = ['SpectralModel']


class SpectralModel:

    def __init__(self, spectral_table=None, **kwargs):
        self.spectral_table = spectral_table

    @property
    def spectral_table(self):
        return self._spectral_table

    @spectral_table.setter
    def spectral_table(self, value):
        if isinstance(value, ndcube.NDCube):
            self._spectral_table = value
        else:
            from synthesizAR.atomic.idl import read_spectral_table
            if value is None:
                value = get_pkg_data_filename('data/chianti-spectrum.asdf',
                                              package='mocksipipeline.physics.spectral')
            self._spectral_table = read_spectral_table(value)

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
        
    @staticmethod
    def build_spectral_table(**kwargs):
        """
        Build the spectral table with some sensible defaults for MOXSI
        """
        from synthesizAR.atomic.idl import compute_spectral_table
        temperature = kwargs.pop('temperature', 10**np.arange(5.5, 7.6, 0.1)*u.K)
        density = kwargs.pop('density', None)
        if density is None:
            # Assume constant pressure
            pressure = 1e15 * u.cm**(-3) * u.K
            density = pressure / temperature
        wave_min = kwargs.pop('wave_min', 0.5 * u.angstrom)
        wave_max = kwargs.pop('wave_max', 60.5 * u.angstrom)
        delta_wave = kwargs.pop('delta_wave', 25 * u.milliangstrom)
        ioneq_filename = kwargs.pop('ioneq_filename', 'chianti.ioneq')
        abundance_filename = kwargs.pop('abundance_filename', 'sun_coronal_1992_feldman.abund')
        ion_list = kwargs.pop('ion_list', None)
        include_continuum = kwargs.pop('include_continuum', True)
        chianti_dir = kwargs.pop('chianti_dir', None)
        return compute_spectral_table(
            temperature,
            density,
            wave_min,
            wave_max,
            delta_wave,
            ioneq_filename,
            abundance_filename,
            ion_list=ion_list,
            include_continuum=include_continuum,
            chianti_dir=chianti_dir,
        )
