"""
Classes for computing wavelength response functions for MOXSI
"""
import copy

import numpy as np
import astropy.units as u
import astropy.constants as const
import astropy.table
from astropy.utils.data import get_pkg_data_filename
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate
from scipy.interpolate import interp1d
from sunpy.util import MetaDict
from sunpy.io.special import read_genx

from overlappy.wcs import overlappogram_fits_wcs, pcij_matrix
from synthesizAR.instruments.util import extend_celestial_wcs

from mocksipipeline.detector.filter import ThinFilmFilter

__all__ = [
    'Channel',
    'SpectrogramChannel',
    'convolve_with_response',
    'get_all_filtergram_channels',
    'get_all_dispersed_channels',
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

    def __init__(self, name, filters=None, full_detector=True):
        self.name = name
        self.filters = filters
        self.full_detector = full_detector

    def __repr__(self):
        return f"""MOXSI Detector Channel--{self.name}
-----------------------------------
Spectral order: {self.spectral_order}
Filter: {self.filter_label}
Detector dimensions: {self.detector_shape}
Wavelength range: [{self.wavelength[0]}, {self.wavelength[-1]}]
Spectral resolution: {self.spectral_resolution}
Spatial resolution: {self.resolution}
Reference pixel: {self.reference_pixel}
"""

    @staticmethod
    def _read_genx_instrument_data(name, instrument_file=None):
        # This is deprecated functionality for pulling parameters from the old genx file
        if instrument_file is None:
            instrument_file = get_pkg_data_filename('data/MOXSI_effarea.genx',
                                                    package='mocksipipeline.detector')
        instrument_data = read_genx(instrument_file)
        index_mapping = {}
        for i, c in enumerate(instrument_data['SAVEGEN0']):
            index_mapping[c['CHANNEL']] = i
        return MetaDict(instrument_data['SAVEGEN0'][index_mapping[name]])

    @property
    def _default_filters(self):
        # These are based off the current instrument design
        polymide = ThinFilmFilter(elements=['C', 'H', 'N', 'O'],
                                  quantities=[22, 10, 2, 5],
                                  density=1.43*u.g/u.cm**3,
                                  thickness=1*u.micron, xrt_table='Chantler')
        aluminum = ThinFilmFilter(elements='Al', thickness=150*u.nm, xrt_table='Chantler')
        return {
            'filtergram_1': ThinFilmFilter('Be', thickness=8*u.micron, xrt_table='Chantler'),
            'filtergram_2': ThinFilmFilter('Be', thickness=30*u.micron, xrt_table='Chantler'),
            'filtergram_3': ThinFilmFilter('Be', thickness=350*u.micron, xrt_table='Chantler'),
            'filtergram_4': [polymide, aluminum],
        }

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, value):
        if value is None:
            value = self._default_filters[self.name]
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
        return (7.4, 7.4) * u.arcsec / u.pixel

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
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    @u.quantity_input
    def wavelength(self) -> u.angstrom:
        # dispersion must start at 0
        wave_min = 0 * u.angstrom
        # this is the maximum wavelength at which emission from
        # the far limb of the sun will still fall on the detector
        wave_max = 90 * u.angstrom
        return np.arange(
            wave_min.to_value('AA'),
            (wave_max + self.spectral_resolution*u.pix).to_value('AA'),
            (self.spectral_resolution*u.pix).to_value('AA'),
        ) * u.angstrom

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
    def pinhole_diameter(self) -> u.cm:
        # NOTE: this number comes from Jake
        return 44 * u.micron

    @property
    @u.quantity_input
    def geometrical_collecting_area(self) -> u.cm**2:
        return np.pi * (self.pinhole_diameter / 2)**2

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
        return u.Quantity(np.ones(self.wavelength.shape))

    @property
    @u.quantity_input
    def detector_efficiency(self) -> u.dimensionless_unscaled:
        # NOTE: this thickness was determined by comparisons between the detector efficiency
        # originally in the genx files from the origional proposal.
        si = ThinFilmFilter('Si', thickness=33*u.micron, xrt_table='Chantler')
        return 1.0 - si.transmissivity(self._energy_no_inf)

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
        return 71.8 * u.milliangstrom / u.pix

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

    def get_wcs(self, observer, roll_angle=90*u.deg, dispersion_angle=0*u.deg):
        pc_matrix = pcij_matrix(roll_angle,
                                dispersion_angle,
                                order=self.spectral_order)
        return overlappogram_fits_wcs(
            self.detector_shape,
            self.wavelength,
            (self.resolution[0], self.resolution[1], self.spectral_resolution),
            reference_pixel=self.reference_pixel,
            reference_coord=(0*u.arcsec, 0*u.arcsec, 0*u.angstrom),
            pc_matrix=pc_matrix,
            observer=observer,
        )


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

    def __init__(self, order, **kwargs):
        self.spectral_order = order
        self.grating_file = kwargs.pop('grating_file', None)
        if self.grating_file is None:
            self.grating_file = get_pkg_data_filename('data/hetgD1996-11-01greffpr001N0007.fits',
                                                      package='mocksipipeline.detector')
        super().__init__('dispersed_image', **kwargs)

    @property
    def _default_filters(self):
        return {
            'dispersed_image': ThinFilmFilter(elements='Al', thickness=150*u.nm, xrt_table='Chantler'),
        }

    @property
    def _reference_pixel_lookup(self):
        lookup = super()._reference_pixel_lookup
        px = (self.detector_shape[1] - 1)/2
        if self.full_detector:
            py = (self.detector_shape[0]/2 - 1)/2
        else:
            py = (self.detector_shape[0] - 1)/2
        lookup['dispersed_image'] = (px, py, 0) * u.pix
        return lookup

    @property
    def spectral_order(self):
        return self._spectral_order

    @spectral_order.setter
    def spectral_order(self, value):
        allowed_spectral_orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        if value not in allowed_spectral_orders:
            raise ValueError(f'{value} is not an allowed spectral order.')
        self._spectral_order = value

    @property
    @u.quantity_input
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        # Get grating efficiency tables for both shells of the HEG grating
        heg_shell_4 = self._read_grating_file(self.grating_file, 3)
        heg_shell_6 = self._read_grating_file(self.grating_file, 4)
        # Interpolate grating efficiency of relevant order to detector wavelength
        ge_shell_4 = self._wavelength_interpolator(heg_shell_4['wavelength'],
                                                   heg_shell_4[f'grating_efficiency_{self.spectral_order}'])
        ge_shell_6 = self._wavelength_interpolator(heg_shell_6['wavelength'],
                                                   heg_shell_6[f'grating_efficiency_{self.spectral_order}'])
        # Shells corresponding to HEG gratings should be equivalent to what we have so averaging
        # the two gives us the best estimate for our grating efficiency
        return u.Quantity((ge_shell_4 + ge_shell_6) / 2)

    def _wavelength_interpolator(self, wavelength, value):
        # Interpolates an array stored in the data file from the tabulated
        # wavelengths to the wavelengths we want for this instrument
        f_interp = interp1d(wavelength.to_value('Angstrom'),
                            value,
                            bounds_error=False,
                            fill_value=0.0)
        return f_interp(self.wavelength.to_value('Angstrom'))

    @staticmethod
    def _read_grating_file(filename, hdu):
        # This function reads the grating efficiency for the HETG Chandra gratings
        # from the calibration FITS files. These were downloaded using the Chandra
        # calibration database https://cxc.cfa.harvard.edu/caldb/about_CALDB/directory.html
        tab = astropy.table.QTable.read(filename, hdu=hdu)
        # Separate all orders into individual columns (-11 to +11, including 0)
        orders = np.arange(-11, 12, 1, dtype=int)
        for i, o in enumerate(orders):
            tab[f'grating_efficiency_{o}'] = tab['EFF'].data[:, i]
        tab.remove_columns(['EFF', 'SYS_MIN'])
        tab.rename_column('ENERGY', 'energy')
        tab['wavelength'] = (const.h * const.c / tab['energy']).to('Angstrom')
        return tab


def get_all_filtergram_channels(**kwargs):
    filtergram_names = [f'filtergram_{i}' for i in range(1, 5)]
    return [Channel(name, **kwargs) for name in filtergram_names]


def get_all_dispersed_channels(**kwargs):
    spectral_orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    return [SpectrogramChannel(order, **kwargs)for order in spectral_orders]
