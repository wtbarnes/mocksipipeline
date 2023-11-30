"""
Classes for computing wavelength response functions for MOXSI
"""
import astropy.table
import astropy.units as u
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from overlappy.wcs import overlappogram_fits_wcs, pcij_matrix
from scipy.interpolate import interp1d
from sunpy.io.special import read_genx
from sunpy.util import MetaDict

from mocksipipeline.detector.filter import ThinFilmFilter

__all__ = [
    'Channel',
    'get_all_filtergram_channels',
    'get_all_dispersed_channels',
]


class Channel:
    """
    Access properties of MOXSI channels and compute wavelength response functions

    Parameters
    ----------
    name: `str`
        Name of the filtergram. This determines the position of the image on the detector.
    order: `int`, optional
        Spectral order for the channel. By default, this is 0.
    filters: `~mocksipipeline.detector.filter.ThinFilmFilter` or list
        If multiple filters are specified, the the filter transmission is computed as
        the product of the transmissivities. If not specified, fall back to default
        filters for a particular channel.
    """

    def __init__(self, name, order=0, filters=None, **kwargs):
        self.name = name
        self.spectral_order = order
        self.xrt_table_name = 'Chantler'  # Intentionally hardcoding this for now
        self.filters = filters
        self.grating_file = kwargs.pop('grating_file', None)
        if self.grating_file is None:
            self.grating_file = get_pkg_data_filename('data/hetgD1996-11-01greffpr001N0007.fits',
                                                      package='mocksipipeline.detector')

    def __repr__(self):
        return f"""MOXSI Detector Channel--{self.name}
-----------------------------------
Spectral order: {self.spectral_order}
Filter: {self.filter_label}
Detector dimensions: {self.detector_shape}
Wavelength range: [{self.wavelength_min}, {self.wavelength_max}]
Spectral resolution: {self.spectral_resolution}
Spatial resolution: {self.resolution}
Reference pixel: {self.reference_pixel}
"""

    @property
    def _default_filters(self):
        # These are based off the current instrument design
        polymide = ThinFilmFilter(elements=['C', 'H', 'N', 'O'],
                                  quantities=[22, 10, 2, 5],
                                  density=1.43*u.g/u.cm**3,
                                  thickness=1*u.micron,
                                  xrt_table=self.xrt_table_name)
        aluminum = ThinFilmFilter(elements='Al', thickness=150*u.nm, xrt_table=self.xrt_table_name)
        return {
            'filtergram_1': ThinFilmFilter('Be', thickness=8*u.micron, xrt_table=self.xrt_table_name),
            'filtergram_2': ThinFilmFilter('Be', thickness=30*u.micron, xrt_table=self.xrt_table_name),
            'filtergram_3': ThinFilmFilter('Be', thickness=350*u.micron, xrt_table=self.xrt_table_name),
            'filtergram_4': [polymide, aluminum],
            'spectrogram_1': aluminum,
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
        return self._spectral_order

    @spectral_order.setter
    def spectral_order(self, value):
        allowed_spectral_orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        if value not in allowed_spectral_orders:
            raise ValueError(f'{value} is not an allowed spectral order.')
        self._spectral_order = value

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
        p_x = margin + (window - 1)/2
        # NOTE: this is the y coordinate of the reference pixel of all of the
        # filtergram images
        p_y = (self.detector_shape[0]/2 - 1)/2 + self.detector_shape[0]/2
        # NOTE: the order here is Cartesian, not (row, column)
        # NOTE: this is 1-indexed
        return {
            'filtergram_1': (p_x, p_y, 0) * u.pix,
            'filtergram_2': (p_x + window, p_y, 0) * u.pix,
            'filtergram_3': (p_x + 2*window, p_y, 0) * u.pix,
            'filtergram_4': (p_x + 3*window, p_y, 0) * u.pix,
            'spectrogram_1': ((self.detector_shape[1] - 1)/2, (self.detector_shape[0]/2 - 1)/2, 0)*u.pix,
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
        # NOTE: The resolution of the wavelength array is adjusted according to the
        # spectral order so that when reprojecting, we do not have gaps in the spectra
        # as the wavelength array gets stretched across the detector
        delta_wave = self.spectral_resolution*u.pix
        delta_wave /= 2 * (1 if self.spectral_order == 0 else np.fabs(self.spectral_order))
        return np.arange(
            self.wavelength_min.to_value('AA'),
            (self.wavelength_max + delta_wave).to_value('AA'),
            delta_wave.to_value('AA'),
        ) * u.angstrom

    @property
    @u.quantity_input
    def wavelength_min(self) -> u.Angstrom:
        return 0 * u.Angstrom  # dispersion must start at 0

    @property
    @u.quantity_input
    def wavelength_max(self) -> u.Angstrom:
        # this is the maximum wavelength at which emission from
        # the far limb of the sun will still fall on the detector
        return 90 * u.Angstrom

    @property
    @u.quantity_input
    def energy(self) -> u.keV:
        return self.wavelength.to('keV', equivalencies=u.spectral())

    @property
    def _energy_out_of_bounds(self):
        # NOTE: This is needed because the functions in xrt called
        # by ThinFilmFilter cannot handle infinite energy but handle
        # NaN fine. The transmissivities at these infinities is just
        # set to 0. This is primarily so that we can handle 0 wavelength.
        # Additionally, there are energy limits imposed by the data used to
        # compute the transmissivity of the materials and thus the energy is
        # set to NaN here as well.
        energy_bounds = {
            'Henke': u.Quantity([10*u.eV, 30*u.keV]),
            'Chantler': u.Quantity([11*u.eV, 405*u.keV]),
            'BrCo': u.Quantity([30*u.eV, 509*u.keV]),
        }
        return np.logical_or(self.energy < energy_bounds[self.xrt_table_name][0],
                             self.energy > energy_bounds[self.xrt_table_name][1])

    @property
    def _energy_no_inf(self):
        # NOTE: This is needed because the functions in xrt called
        # by ThinFilmFilter cannot handle infinite energy but handle
        # NaN fine. The transmissivities at these infinities is just
        # set to 0. This is primarily so that we can handle 0 wavelength.
        return np.where(self._energy_out_of_bounds, np.nan, self.energy)

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
        # NOTE: Filtergrams do not have any grating in front of them
        if 'filtergram' in self.name:
            return u.Quantity(np.ones(self.wavelength.shape))
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
        tab['wavelength'] = tab['energy'].to('Angstrom', equivalencies=u.spectral())
        return tab

    @property
    @u.quantity_input
    def quantum_efficiency(self) -> u.dimensionless_unscaled:
        # NOTE: Value of 10 microns based on conversations with A. Caspi regarding
        # expected width. This thickness is different from what was used for calculating
        # the detector efficiency in the original CubIXSS proposal.
        si = ThinFilmFilter('Si', thickness=10*u.micron, xrt_table=self.xrt_table_name)
        return 1.0 - si.transmissivity(self._energy_no_inf)

    @property
    @u.quantity_input
    def effective_area(self) -> u.cm**2:
        effective_area = (self.geometrical_collecting_area *
                          self.filter_transmission *
                          self.quantum_efficiency *
                          self.grating_efficiency)
        return np.where(self._energy_out_of_bounds, 0*u.cm**2, effective_area)

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
        return u.Quantity(1.8, 'ct / electron')

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
        wave_response = (self.effective_area *
                         self.electron_per_photon *
                         self.camera_gain)
        return np.where(self._energy_out_of_bounds, 0*u.Unit('cm2 ct /ph'), wave_response)

    def get_wcs(self, observer, roll_angle=90*u.deg, dispersion_angle=0*u.deg):
        pc_matrix = pcij_matrix(roll_angle,
                                dispersion_angle,
                                order=self.spectral_order)
        # NOTE: This is calculated explicitly here because the wavelength array for a
        # particular channel is not necessarily have a resolution equal to that of
        # the spectral plate scale.
        # NOTE: Calculated based on the information in the numpy docs for arange
        # https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        wave_step = self.spectral_resolution * u.pix
        wave_interval = self.wavelength_max - self.wavelength_min
        wave_shape = (np.ceil((wave_interval / wave_step).decompose().value).astype(int),)
        return overlappogram_fits_wcs(
            wave_shape+self.detector_shape,
            (self.resolution[0], self.resolution[1], self.spectral_resolution),
            reference_pixel=self.reference_pixel,
            reference_coord=(0*u.arcsec, 0*u.arcsec, 0*u.angstrom),
            pc_matrix=pc_matrix,
            observer=observer,
        )


def get_all_filtergram_channels(**kwargs):
    filtergram_names = [f'filtergram_{i}' for i in range(1, 5)]
    return [Channel(name, order=0, **kwargs) for name in filtergram_names]


def get_all_dispersed_channels(**kwargs):
    spectral_orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    return [Channel('spectrogram_1', order, **kwargs) for order in spectral_orders]
