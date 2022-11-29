"""
Classes for computing wavelength response functions for MOXSI
"""
import copy

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.utils.data import get_pkg_data_filename
from ndcube import NDCube
from scipy.interpolate import interp1d
from sunpy.util import MetaDict
from sunpy.io.special import read_genx

from synthesizAR.instruments.util import extend_celestial_wcs

__all__ = [
    'Channel',
    'SpectrogramChannel',
    'convolve_with_response',
]


def convolve_with_response(cube, channel, include_gain=True):
    """
    Convolve spectral cube with wavelength response to convert spectra to instrument units.
    
    Parameters
    ----------
    cube : `ndcube.NDCube`
    channel : `Channel`
    include_gain : `bool`, optional
        If True, include conversion factor from photons to DN
    
    Return
    ------
    : `ndcube.NDCube`
        Spectral cube in detector units convolved with instrument response
    """
    # Compute instrument response
    if include_gain:
        response = channel.wavelength_response
    else:
        response = channel.effective_area
    # Multiply by the spatial and spectral plate scale (factor of sr)
    # NOTE: does it make sense to do this before interpolating to the *exact*
    # instrument resolution?
    response *= channel.plate_scale
    response *= channel.spectral_resolution * (1 * u.pix)

    # Interpolate spectral cube to the wavelength array of the channel
    # FIXME: In cases where we are using a binned spectra,
    # we should be *rebinning* the spectral cube, not just interpolating it.
    # The spectra should be rebinned to the range and bin width of the
    # instrument.
    # NOTE: it is ok that our spectral cube is not necessarily guaranteed
    # to be at the spectral plate scale of the instrument as we will reproject
    # to the correct spectral plate scale at a later point in the pipeline
    f_response = interp1d(cube.axis_world_coords(0)[0].to_value('Angstrom'),
                          cube.data,
                          axis=0,
                          bounds_error=False,
                          fill_value=0.0,)  # Response is 0 outside of the response range
    data_interp = f_response(channel.wavelength.to_value('Angstrom'))
    data_interp *= response.to_value()[:, np.newaxis, np.newaxis]
    
    unit = cube.unit * response.unit
    meta = copy.deepcopy(cube.meta)
    meta['CHANNAME'] = channel.name
    # Reset the units if they were in the metadata
    meta.pop('BUNIT', None)

    # Construct new WCS for the modified wavelength axis
    new_wcs = extend_celestial_wcs(cube[0].wcs.low_level_wcs,
                                   channel.wavelength,
                                   'wavelength',
                                   'em.wl')
    
    return NDCube(data_interp, wcs=new_wcs, meta=meta, unit=unit)


class Channel:

    def __init__(self, name, instrument_file=None, filter=None):
        # Switch this to accept a filter type or an order and then construct name
        # based on that.
        self._name = name
        if instrument_file is None:
            instrument_file = get_pkg_data_filename('data/MOXSI_effarea.genx', package='mocksipipeline.detector')
        self._instrument_data = self._get_instrument_data(instrument_file)
        self.filter = filter
        
    def _get_instrument_data(self, instrument_file):
        return read_genx(instrument_file)

    @property
    @u.quantity_input
    def resolution(self) -> u.Unit('arcsec / pix'):
        # These numbers come from Jake and Albert / CSR
        # the order is lon, lat
        # TODO: this should probably go in a file somewhere
        # We could pull this from the 'PIX_SIZE' key in the data files, but it
        # appears that value may be outdated
        return (5.66, 5.66) * u.arcsec / u.pixel

    @property
    def detector_shape(self):
        return (750, 2000)
        
    @property
    def _data(self):
        index_mapping = {}
        for i,c in enumerate(self._instrument_data['SAVEGEN0']):
            index_mapping[c['CHANNEL']] = i
        return MetaDict(self._instrument_data['SAVEGEN0'][index_mapping[self._name]])
        
    @property
    def name(self):
        return self._name
        
    @property
    @u.quantity_input
    def wavelength(self) -> u.angstrom:
        return u.Quantity(self._data['wave'], 'angstrom')
    
    @property
    @u.quantity_input
    def energy(self) -> u.keV:
        return const.h * const.c / self.wavelength
        
    @property
    @u.quantity_input
    def geometrical_collecting_area(self) -> u.cm**2:
        return u.Quantity(self._data['geo_area'], 'cm^2')
        
    @property
    @u.quantity_input
    def filter_transmission(self) -> u.dimensionless_unscaled:
        if self.filter is None:
            return u.Quantity(self._data['filter'])
        else:
            return self.filter.transmissivity(self.energy)
        
    @property
    @u.quantity_input
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        # NOTE: this is just 1 for the filtergrams
        return u.Quantity(self._data['grating'])
        
    @property
    @u.quantity_input
    def detector_efficiency(self) -> u.dimensionless_unscaled:
        return u.Quantity(self._data['det'])
    
    @property
    @u.quantity_input
    def effective_area(self) -> u.cm**2:
        return (self.geometrical_collecting_area * 
                self.filter_transmission *
                self.grating_efficiency *
                self.detector_efficiency)
    
    @property
    @u.quantity_input
    def plate_scale(self) -> u.steradian / u.pixel:
        """
        This is the solid angle per pixel
        """
        area = (self.resolution[0] * u.pix) * (self.resolution[1] * u.pix)
        return area / u.pix

    @property
    @u.quantity_input
    def spectral_resolution(self) -> u.Unit('Angstrom / pix'):
        return 55 * u.milliangstrom / u.pix

    @property
    def spectral_order(self):
        return 0
    
    @property
    @u.quantity_input
    def gain(self) -> u.ct / u.photon:
        # TODO: double check the units on this
        camera_gain = u.Quantity(self._data['gain'], 'ct / electron')
        # This is approximately the average energy to free an electron
        # in silicon
        energy_per_electron = 3.65 * u.Unit('eV / electron')
        energy_per_photon = const.h * const.c / self.wavelength / u.photon
        electron_per_photon = energy_per_photon / energy_per_electron
        # Cannot discharge less than one electron per photon
        discharge_floor = 1 * u.Unit('electron / photon')
        electron_per_photon[electron_per_photon<discharge_floor] = discharge_floor
        return electron_per_photon * camera_gain

    @property
    def wavelength_response(self) -> u.Unit('cm^2 ct / photon'):
        return self.effective_area * self.gain
    

class SpectrogramChannel(Channel):
    
    def __init__(self, order, **kwargs):
        self._spectral_order = order
        name = f'MOXSI_S{int(np.fabs(order))}'
        super().__init__(name, **kwargs)

    @property
    def spectral_order(self):
        return self._spectral_order
