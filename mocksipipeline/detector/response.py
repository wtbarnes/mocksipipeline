"""
Classes for computing wavelength response functions for MOXSI
"""
import copy

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.utils.data import get_pkg_data_filename
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate
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


def convolve_with_response(cube, channel, electrons=True, include_gain=False):
    """
    Convolve spectral cube with wavelength response to convert spectra to instrument units.

    Parameters
    ----------
    cube : `ndcube.NDCube`
    channel : `Channel`
    electrons : `bool`, optional
        If True, include conversion factor from photons to electrons.
    include_gain : `bool`, optional
        If True, include conversion fractor from electrons to DN. Cannot be true
        if `electrons` is False.

    Return
    ------
    : `ndcube.NDCube`
        Spectral cube in detector units convolved with instrument response
    """
    # Compute instrument response
    response = channel.effective_area
    if electrons:
        response *= channel.electron_per_photon
        if include_gain:
            response *= channel.camera_gain
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
    # NOTE: this is only ok if we are reprojecting the full cube to the full
    # overlappogram in one go. With the per slice approach, the wavelength
    # grids must be the same, i.e. the wavelength grid exposed by the channel
    # will be the wavelength grid used by overlappogram in the wavelength axis
    f_response = interp1d(cube.axis_world_coords(0)[0].to_value('Angstrom'),
                          cube.data,
                          axis=0,
                          bounds_error=False,
                          fill_value=0.0,)  # Response is 0 outside of the response range
    data_interp = f_response(channel.wavelength.to_value('Angstrom'))
    data_interp = (data_interp.T * response.to_value()).T

    unit = cube.unit * response.unit
    meta = copy.deepcopy(cube.meta)
    meta['CHANNAME'] = channel.name
    # Reset the units if they were in the metadata
    meta.pop('BUNIT', None)

    # Construct new WCS for the modified wavelength axis
    # NOTE: When there is only one axis that corresponds to wavlength, then
    # just dconstruct a new wavelength axis
    if len(cube.data.shape) == 1:
        new_wcs = QuantityTableCoordinate(channel.wavelength,
                                          names='wavelength',
                                          physical_types='em.wl').wcs
    else:
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
        Name of the filtergram. This determines the position of the image on the detector.
    filter: `~mocksipipeline.detector.filter.ThinFilmFilter` or list
        If multiple filters are specified, the the filter transmission is computed as
        the product of the transmissivities.
    instrument_file: `str`, optional
        Instrument file (in genx format) to pull wavelength response information from.
        This is mostly used for getting the grating and detector efficiency.
    full_detector: `bool`, optional
        If True (default), the detector shape and reference pixel include the full
        detector, including the the halves for the disersed image and filtergram images.
        If False, the reference pixel and detector shape denote only the half where the
        relevant half of the detector.
    """

    def __init__(self, name, filters, instrument_file=None, full_detector=True):
        self.name = name
        self.filters = filters
        if instrument_file is None:
            instrument_file = get_pkg_data_filename('data/MOXSI_effarea.genx', package='mocksipipeline.detector')
        self._instrument_data = self._get_instrument_data(instrument_file)
        self.full_detector = full_detector

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
        if self.full_detector:
            return (1500, 2000)
        else:
            return (750, 2000)

    @property
    def _reference_pixel_lookup(self):
        # NOTE: this is the number of pixels between the edge of the detector
        # and the leftmost and rightmost filtergram images
        margin = 50
        # NOTE: this is the width, in pixel space, of each filtergram image
        window = 475
        # NOTE: this is the x coordinate of the reference pixel of the leftmost
        # filtergram image
        p_x = margin + (window - 1)/2
        # NOTE: this is the y coordinate of the reference pixel of all of the
        # filtergram images
        if self.full_detector:
            p_y = (self.detector_shape[0]/2 - 1)/2 + self.detector_shape[0]/2
        else:
            p_y = (self.detector_shape[0] - 1)/2
        # NOTE: the order here is Cartesian, not (row, column)
        # NOTE: this is 1-indexed
        return {
            'filtergram_1': (p_x, p_y, 0) * u.pix,
            'filtergram_2': (p_x + window, p_y, 0) * u.pix,
            'filtergram_3': (p_x + 2*window, p_y, 0) * u.pix,
            'filtergram_4': (p_x + 3*window, p_y, 0) * u.pix,
        }

    @property
    @u.quantity_input
    def reference_pixel(self) -> u.pixel:
        return self._reference_pixel_lookup[self.name]

    @property
    def _index_mapping(self):
        index_mapping = {}
        for i, c in enumerate(self._instrument_data['SAVEGEN0']):
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
    def _wavelength_data(self) -> u.angstrom:
        return u.Quantity(self._data['wave'], 'angstrom')

    @property
    @u.quantity_input
    def wavelength(self) -> u.angstrom:
        # dispersion must start at 0
        wave_min = 0 * u.angstrom
        # this is the maximum wavelength at which emission from
        # the far limb of the sun will still fall on the detector
        wave_max = 68 * u.angstrom
        return np.arange(
            wave_min.to_value('AA'),
            (wave_max + self.spectral_resolution*u.pix).to_value('AA'),
            (self.spectral_resolution*u.pix).to_value('AA'),
        ) * u.angstrom

    def _wavelength_interpolator(self, value):
        # Interpolates an array stored in the data file from the tabulated
        # wavelengths to the wavelengths we want for this instrument
        # NOTE: We are purposefully extrapolating as we assume that the difference
        # between the two arrays is relatively small.
        # NOTE: Once we can calculate each quantity ourselves, this can be removed
        f_interp = interp1d(self._wavelength_data.to_value('Angstrom'),
                            value,
                            fill_value='extrapolate')
        value_interp = f_interp(self.wavelength.to_value('Angstrom'))
        # NOTE: setting anything less than the minimum wavelength to 0 as extrapolation
        # at these wavelengths/energies is difficult. Extrapolating at low wavelengths
        # seems to be ok.
        return np.where(self.wavelength < self._wavelength_data[0], 0.0, value_interp)

    @property
    @u.quantity_input
    def energy(self) -> u.keV:
        return const.h * const.c / self.wavelength

    @property
    def _energy_is_inf(self):
        # NOTE: This is needed becuase the functions in xrt called
        # by ThinFilmFilter cannot handle infinite energy but handle
        # NaN fine. The transmissivities at these infinities is just
        # set to 0. This is primarily so that we can handle 0 wavelength.
        return np.isinf(self.energy)

    @property
    def _energy_no_inf(self):
        # NOTE: This is needed becuase the functions in xrt called
        # by ThinFilmFilter cannot handle infinite energy but handle
        # NaN fine. The transmissivities at these infinities is just
        # set to 0. This is primarily so that we can handle 0 wavelength.
        return np.where(self._energy_is_inf, np.nan, self.energy)

    @property
    @u.quantity_input
    def geometrical_collecting_area(self) -> u.cm**2:
        return u.Quantity(self._data['geo_area'], 'cm^2')

    @property
    @u.quantity_input
    def filter_transmission(self) -> u.dimensionless_unscaled:
        ft = u.Quantity(np.ones(self.energy.shape))
        for f in self.filters:
            ft *= f.transmissivity(self._energy_no_inf)
        return ft

    @property
    @u.quantity_input
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        # NOTE: this is just 1 for the filtergrams
        return u.Quantity(self._wavelength_interpolator(self._data['grating']))

    @property
    @u.quantity_input
    def detector_efficiency(self) -> u.dimensionless_unscaled:
        return u.Quantity(self._wavelength_interpolator(self._data['det']))

    @property
    @u.quantity_input
    def effective_area(self) -> u.cm**2:
        effective_area = (self.geometrical_collecting_area *
                          self.filter_transmission *
                          self.grating_efficiency *
                          self.detector_efficiency)
        return np.where(self._energy_is_inf, 0*u.cm**2, effective_area)

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
    @u.quantity_input
    def camera_gain(self) -> u.Unit('ct / electron'):
        # TODO: double check the units on this
        return u.Quantity(1.0, 'ct / electron')

    @property
    @u.quantity_input
    def electron_per_photon(self) -> u.electron / u.photon:
        # This is approximately the average energy to free an electron
        # in silicon
        energy_per_electron = 3.65 * u.Unit('eV / electron')
        energy_per_photon = self.energy / u.photon
        electron_per_photon = energy_per_photon / energy_per_electron
        return electron_per_photon

    @property
    @u.quantity_input
    def wavelength_response(self) -> u.Unit('cm^2 ct / photon'):
        wave_response = self.effective_area * self.electron_per_photon * self.camera_gain
        return np.where(self._energy_is_inf, 0*u.Unit('cm2 ct /ph'), wave_response)


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
        px = (self.detector_shape[1] - 1)/2
        if self.full_detector:
            py = (self.detector_shape[0]/2 - 1)/2
        else:
            py = (self.detector_shape[0] - 1)/2
        lookup['dispersed_image'] = (px, py, 0)*u.pix
        return lookup

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
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        ge = super().grating_efficiency
        if self.include_au_cr:
            au_layer = ThinFilmFilter(elements='Au', thickness=20*u.nm, xrt_table='Chantler')
            cr_layer = ThinFilmFilter(elements='Cr', thickness=5*u.nm, xrt_table='Chantler')
            ge *= au_layer.transmissivity(self._energy_no_inf)
            ge *= cr_layer.transmissivity(self._energy_no_inf)
        return ge
