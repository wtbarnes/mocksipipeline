"""
Classes for computing wavelength response functions for MOXSI
"""
from functools import cached_property

import astropy.table
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename
from overlappy.wcs import overlappogram_fits_wcs, pcij_matrix
from scipy.interpolate import interp1d
from sunpy.coordinates import get_earth
from sunpy.map import solar_angular_radius

from mocksipipeline.instrument.optics.filter import ThinFilmFilter

__all__ = [
    'Channel',
]


class Channel:
    """
    Access properties of MOXSI channels and compute wavelength response functions

    Parameters
    ----------
    name: `str`
        Name of the channel. If this is a filtergram channel, the word "filtergram"
        should appear somewhere in it. This is used to set the grating efficiency
        to 1 since the filtergram images do not pass through the dispersive grating.
    filters: `~mocksipipeline.detector.filter.ThinFilmFilter` or list, optional
        If multiple filters are specified, the filter transmission is computed as
        the product of the transmissivities. If not specified, fall back to default
        filters for a particular channel.
    order: `int`, optional
        Spectral order for the channel. By default, this is 0.
    design: `~mocksipipeline.detector.design.OpticalDesign`, optional
        Instrument optical design
    aperture: `~mocksipipeline.instrument.optics.aperture.AbstractAperture`
        Aperture model used for calculating geometrical area
    reference_pixel: `tuple`,
        Pixel location of solar image in cartesian coordinates
    """

    def __init__(self, name, filters, order, design, aperture, reference_pixel, **kwargs):
        self.name = name
        self.spectral_order = order
        self.design = design
        self.aperture = aperture
        self.xrt_table_name = 'Chantler'  # Intentionally hardcoding this for now
        self.filters = filters
        self.grating_file = kwargs.pop('grating_file', None)
        if self.grating_file is None:
            self.grating_file = get_pkg_data_filename('data/hetgD1996-11-01greffpr001N0007.fits',
                                                      package='mocksipipeline.instrument.optics')
        self.reference_pixel = reference_pixel

    def __repr__(self):
        return f"""MOXSI Detector Channel--{self.name}
-----------------------------------
Spectral order: {self.spectral_order}
Filter: {self.filter_label}
Detector dimensions: {self.detector_shape}
Wavelength range: [{self.wavelength_min}, {self.wavelength_max}]
Spectral plate scale: {self.spectral_plate_scale}
Spatial plate scale: {self.spatial_plate_scale}
Reference pixel: {self.reference_pixel}
aperture: {self.aperture}
"""

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
        return self._spectral_order

    @spectral_order.setter
    def spectral_order(self, value):
        allowed_spectral_orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        if value not in allowed_spectral_orders:
            raise ValueError(f'{value} is not an allowed spectral order.')
        self._spectral_order = value

    @property
    @u.quantity_input
    def spatial_plate_scale(self) -> u.Unit('arcsec / pix'):
        return self.design.spatial_plate_scale

    @property
    @u.quantity_input
    def pixel_solid_angle(self) -> u.Unit('steradian / pix'):
        return self.design.pixel_solid_angle

    @property
    @u.quantity_input
    def spectral_plate_scale(self) -> u.Unit('Angstrom / pix'):
        return self.design.spectral_plate_scale

    @property
    def detector_shape(self):
        return self.design.detector_shape

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    @u.quantity_input
    def wavelength(self) -> u.angstrom:
        delta_wave = self.spectral_plate_scale * u.pix
        if self.spectral_order != 0:
            # NOTE: The resolution of the wavelength array is adjusted according to the
            # spectral order so that when reprojecting, we do not have gaps in the spectra
            # as the wavelength array gets stretched across the detector in the case of the
            # dispersed images.
            delta_wave /= 2 * np.fabs(self.spectral_order)
        return np.arange(
            self.wavelength_min.to_value('AA'),
            (self.wavelength_max + delta_wave).to_value('AA'),
            delta_wave.to_value('AA'),
        ) * u.angstrom

    @property
    @u.quantity_input
    def wavelength_min(self) -> u.Angstrom:
        return 0 * u.Angstrom  # dispersion must start at 0

    @cached_property
    @u.quantity_input
    def wavelength_max(self) -> u.Angstrom:
        # Estimate the maximum wavelength that would fall on the detector
        # by assuming that the farthest a source location could be is off
        # limb by 25% of the solar radius. The reason for doing this per
        # spectral order is to minimize the number of wavelengths that are
        # projected completely off the detector.
        obs = get_earth('2020-01-01')
        origin = SkyCoord(Tx=0*u.arcsec, Ty=0*u.arcsec, frame='helioprojective', observer=obs)
        # NOTE: This assumes the origin falls in the middle of the detector
        width_pixel = (self.detector_shape[1] / 2)* u.pix
        width_pixel += 1.25 * solar_angular_radius(origin) / self.spatial_plate_scale[0]
        wave_max = width_pixel * self.spectral_plate_scale
        if self.spectral_order != 0:
            wave_max /= np.fabs(self.spectral_order)
        return wave_max

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
            'Henke': u.Quantity([10 * u.eV, 30 * u.keV]),
            'Chantler': u.Quantity([11 * u.eV, 405 * u.keV]),
            'BrCo': u.Quantity([30 * u.eV, 509 * u.keV]),
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
    def geometrical_collecting_area(self) -> u.cm ** 2:
        return self.aperture.area

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
        r"""
        The quantum efficiency is computed as the transmittance of
        :math:`\mathrm{SiO}_2` multiplied by the absorption of Si.
        """
        # NOTE: Value of 10 microns based on conversations with A. Caspi regarding
        # expected width. This thickness is different from what was used for calculating
        # the detector efficiency in the original CubIXSS proposal.
        si = ThinFilmFilter('Si', thickness=10 * u.micron, xrt_table=self.xrt_table_name)
        sio2 = ThinFilmFilter(['Si', 'O'], thickness=50 * u.AA, quantities=[1, 2], xrt_table=self.xrt_table_name)
        return sio2.transmissivity(self._energy_no_inf) * (1.0 - si.transmissivity(self._energy_no_inf))

    @property
    @u.quantity_input
    def effective_area(self) -> u.cm ** 2:
        effective_area = (self.geometrical_collecting_area *
                          self.filter_transmission *
                          self.quantum_efficiency *
                          self.grating_efficiency)
        return np.where(self._energy_out_of_bounds, 0 * u.cm ** 2, effective_area)

    @property
    @u.quantity_input
    def camera_gain(self) -> u.Unit('ct / electron'):
        return self.design.camera_gain

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
        return np.where(self._energy_out_of_bounds, 0 * u.Unit('cm2 ct /ph'), wave_response)

    def get_wcs(self, observer, roll_angle=90 * u.deg):
        pc_matrix = pcij_matrix(roll_angle,
                                self.design.grating_roll_angle,
                                order=self.spectral_order)
        # NOTE: This is calculated explicitly here because the wavelength array for a
        # particular channel does not necessarily have a resolution equal to that of
        # the spectral plate scale.
        # NOTE: Calculated based on the information in the numpy docs for arange
        # https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        wave_step = self.spectral_plate_scale * u.pix
        wave_interval = self.wavelength_max - self.wavelength_min
        wave_shape = (np.ceil((wave_interval / wave_step).decompose().value).astype(int),)
        return overlappogram_fits_wcs(
            wave_shape + self.detector_shape,
            (self.spatial_plate_scale[0], self.spatial_plate_scale[1], self.spectral_plate_scale),
            reference_pixel=self.reference_pixel,
            reference_coord=(0 * u.arcsec, 0 * u.arcsec, 0 * u.angstrom),
            pc_matrix=pc_matrix,
            observer=observer,
        )
