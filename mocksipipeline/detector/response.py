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

from mocksipipeline.detector.filter import ThinFilmFilter

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
    """
    Access properties of MOXSI channels and compute wavelength response functions

    Parameters
    ----------
    name: `str`
        Name of the filtergram. This determines the position of the image on 
    filter: `~mocksipipeline.detector.filter.ThinFilmFilter` or list
        If multiple filters are specified, the the filter transmission is computed as
        the product of the transmissivities. 
    instrument_file
    """

    def __init__(self, name, filters, instrument_file=None):
        self.name = name
        self.filters = filters
        if instrument_file is None:
            instrument_file = get_pkg_data_filename('data/MOXSI_effarea.genx', package='mocksipipeline.detector')
        self._instrument_data = self._get_instrument_data(instrument_file)
        
    def _get_instrument_data(self, instrument_file):
        return read_genx(instrument_file)
    
    @property
    def filters(self):
        return self._filters
    
    @filters.setter
    def filters(self, value):
        if isinstance(value, ThinFilmFilter):
            self._filters = [value]
        elif isinstance(value, list):
            self._filters = value
        else:
            raise ValueError(f'{type(value)} is not a supported type for filters.')

    @property
    def filter_label(self):
        label = '+'.join([f.chemical_formula for f in self.filters])
        thickness = u.Quantity([f.thickness for f in self.filters]).sum()
        label += f', {thickness:.3f}'
        return label

    @property
    def spectral_order(self):
        return 0

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
        # NOTE: this is the full detector, including both the filtergrams and
        # the dispersed image
        # NOTE: the order here is (number of rows, number of columns)
        return (1500, 2000)

    @property
    def _reference_pixel_lookup(self):
        # NOTE: this is the number of pixels between the edge of the detector
        # and the leftmost and rightmost filtergram images
        margin = 50
        # NOTE: this is the width, in pixel space, of each filtergram image
        window = 475
        # NOTE: this is the x coordinate of the reference pixel of the leftmost
        # filtergram image
        p_x = margin + (window + 1)/2
        # NOTE: this is the y coordinate of the reference pixel of all of the
        # filtergram images
        p_y = (self.detector_shape[0]/2 + 1)/2 + self.detector_shape[0]/2
        # NOTE: the order here is Cartesian, not (row, column)
        # NOTE: this is 1-indexed
        return {
            'filtergram_1': (p_x, p_y, 1) * u.pix,
            'filtergram_2': (p_x + window, p_y, 1) * u.pix,
            'filtergram_3': (p_x + 2*window, p_y, 1) * u.pix,
            'filtergram_4': (p_x + 3*window, p_y, 1) * u.pix,
        }

    @property
    @u.quantity_input
    def reference_pixel(self) -> u.pixel:
        return self._reference_pixel_lookup[self.name]
    
    @property
    def _index_mapping(self):
        index_mapping = {}
        for i,c in enumerate(self._instrument_data['SAVEGEN0']):
            index_mapping[c['CHANNEL']] = i
        return index_mapping
    
    @property
    def _data_index(self):
        # NOTE: this is hardcoded because none of the properties, except the
        # transmission, vary across the channel for the pinhole images
        return self._index_mapping['Be_thin']
        
    @property
    def _data(self):
        return MetaDict(self._instrument_data['SAVEGEN0'][self._data_index])
        
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = value
        
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
        ft = u.Quantity(np.ones(self.energy.shape))
        for f in self.filters:
            ft *= f.transmissivity(self.energy)
        return ft
        
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
        return self._spectral_order

    @spectral_order.setter
    def spectral_order(self, value):
        allowed_spectral_orders = [-3, -1, 0, 1, 3]
        if value not in allowed_spectral_orders:
            raise ValueError(f'{value} is not an allowed spectral order.')
        self._spectral_order = value
    
    @property
    @u.quantity_input
    def camera_gain(self) -> u.Unit('ct / electron'):
        # TODO: double check the units on this
        return u.Quantity(1.0, 'ct / electron')
    
    @property
    @u.quantity_input
    def gain(self) -> u.ct / u.photon:
        # This is approximately the average energy to free an electron
        # in silicon
        energy_per_electron = 3.65 * u.Unit('eV / electron')
        energy_per_photon = const.h * const.c / self.wavelength / u.photon
        electron_per_photon = energy_per_photon / energy_per_electron
        # Cannot discharge less than one electron per photon
        discharge_floor = 1 * u.Unit('electron / photon')
        electron_per_photon[electron_per_photon<discharge_floor] = discharge_floor
        return electron_per_photon * self.camera_gain

    @property
    @u.quantity_input
    def wavelength_response(self) -> u.Unit('cm^2 ct / photon'):
        return self.effective_area * self.gain
    

class SpectrogramChannel(Channel):
    """
    Access properties and compute wavelength responses for the dispersed image
    components.

    Parameters
    ----------
    order
    filter
    include_au_cr
    """
    
    def __init__(self, order, filter, **kwargs):
        self.include_au_cr = kwargs.pop('include_au_cr', True)
        self.spectral_order = order
        super().__init__('dispersed_image', filter, **kwargs)
    
    @property
    def _data_index(self):
        key = f'MOXSI_S{int(np.fabs(self.spectral_order))}'
        return self._index_mapping[key]

    @property
    def _reference_pixel_lookup(self):
        lookup = super()._reference_pixel_lookup
        lookup['dispersed_image'] = ((self.detector_shape[1] + 1)/2,
                                     (self.detector_shape[0]/2 + 1)/2,
                                     1)*u.pix
        return lookup

    @property
    def spectral_order(self):
        return self._spectral_order

    @spectral_order.setter
    def spectral_order(self, value):
        self._spectral_order = value

    @property
    @u.quantity_input
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        ge = super().grating_efficiency
        if self.include_au_cr:
            au_layer = ThinFilmFilter(elements='Au', thickness=20*u.nm)
            cr_layer = ThinFilmFilter(elements='Cr', thickness=5*u.nm)
            ge *= au_layer.transmissivity(self.energy)
            ge *= cr_layer.transmissivity(self.energy)
        return ge

