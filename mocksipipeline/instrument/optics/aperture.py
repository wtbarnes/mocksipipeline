"""
Classes for configuring MOXSI apertures
"""
import abc
import pathlib

import astropy.units as u
import numpy as np
import scipy
import xarray
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_path

__all__ = ['SlotAperture', 'CircularAperture']


def _wfe2psf(wfe, oversample=8):
    """
    Adaptation of CCK Fourier optics code originally in IDL for converting wavefront errors into PSFs
    """
    shp = wfe.shape[0]  # probably assumes it's square, could be changed
    waves = np.exp(2 * np.pi * wfe * 1j)
    big_waves = np.zeros((shp * oversample, shp * oversample), dtype='complex')
    big_waves[0:shp, 0:shp] = waves
    psf = scipy.fft.fft2(big_waves, workers=-1)
    psf = np.abs(psf) ** 2
    psf /= psf.max()
    psf = np.roll(psf, (shp // 2, shp // 2), axis=[0, 1])
    psf = psf[0:shp, 0:shp]
    return psf


class AbstractAperture(abc.ABC):
    """
    An abstract base class for defining the geometry of a MOXSI aperture
    """

    @property
    @abc.abstractmethod
    def name(self):
        ...

    @property
    @abc.abstractmethod
    @u.quantity_input
    def area(self) -> u.cm ** 2:
        ...

    @property
    @abc.abstractmethod
    def mask(self):
        ...

    @u.quantity_input(wavelength=u.Angstrom, resolution=u.m)
    def calculate_psf(self, wavelength, optical_design, resolution=1/8*u.micron, fft_oversample=4, save_psf=False):
        """
        Calculate wavelength-dependent PSF for a given optical design.

        Parameters
        ----------
        wavelength
        optical_design
        resolution
        fft_oversample
        save_psf
        """
        # TODO: Add an option to read from a file--could save the file using the name of the optical design
        # and the name of the aperture
        wavelength = np.atleast_1d(wavelength)
        mask = self.mask(resolution=resolution)
        x = u.Quantity(mask.coords['x'].data, mask.coords['x'].attrs['unit'])
        y = u.Quantity(mask.coords['y'].data, mask.coords['y'].attrs['unit'])
        x_step = x[0, 1] - x[0, 0]
        y_step = y[1, 0] - y[0, 0]
        x_domain = x[0, -1] - x[0, 0]
        f_number = (optical_design.focal_length/x_domain).to_value(u.dimensionless_unscaled)
        # NOTE: The logic here is to multiple the wavefront errors by a large imaginary
        # number so that wavefront errors can easily be tracked as they propagate through
        # the system.
        # NOTE: only the fresnel term depends on wavelength
        wfe = mask.data * 1e9 * 1j
        psfs = []
        for wave in wavelength:
            dx = wave * f_number / fft_oversample
            fresnel_wfe = (x**2 + y**2)/(2*wave*optical_design.focal_length) + 0.j
            wfe_total = wfe + fresnel_wfe.to_value(u.dimensionless_unscaled)
            fresnel_psf = _wfe2psf(wfe_total, oversample=fft_oversample)
            x_psf = (x / x_step * dx).to(x.unit)
            y_psf = (y / y_step * dx).to(x.unit)
            x_coord = xarray.DataArray(data=x_psf[0,:].value,
                                       dims=['x'],
                                       attrs={'unit': x_psf.unit.to_string(format='fits')})
            y_coord = xarray.DataArray(data=y_psf[:,0].value,
                                       dims=['y'],
                                       attrs={'unit': y_psf.unit.to_string(format='fits')})
            _psf = xarray.DataArray(data=fresnel_psf.T,
                                    dims=['x', 'y'],
                                    coords={'x': x_coord, 'y': y_coord})
            psfs.append(_psf)

        # NOTE: Interpolate to the DataArray corresponding to the smallest wavelength
        # because it will have the highest resolution
        psf_target = psfs[np.argmin(wavelength)]
        wave_coord = xarray.DataArray(wavelength.value,
                                      dims=['wavelength'],
                                      attrs={'unit': wavelength.unit.to_string(format='fits')})
        pixel_size_x = optical_design.pixel_size_x.to_value(psf_target.coords['x'].attrs['unit'])
        pixel_size_y = optical_design.pixel_size_y.to_value(psf_target.coords['y'].attrs['unit'])
        pixel_coord_x = psf_target.coords['x'] / pixel_size_x
        pixel_coord_x.attrs['unit'] = 'pixel'
        pixel_coord_y = psf_target.coords['y'] / pixel_size_y
        pixel_coord_y.attrs['unit'] = 'pixel'
        psf_cube = xarray.DataArray(
            data=np.array([psf.interp_like(psf_target) for psf in psfs]),
            dims=['wavelength', 'x', 'y'],
            coords={'x': psf_target.coords['x'],
                    'y': psf_target.coords['y'],
                    'delta_pixel_x': pixel_coord_x,
                    'delta_pixel_y': pixel_coord_y,
                    'wavelength': wave_coord},
        )
        if save_psf:
            data_dir = pathlib.Path(get_pkg_data_path('data', package='mocksipipeline.instrument.optics'))
            psf_cube.to_netcdf(data_dir / f'psf_{optical_design.name}_{self.name}.nc')

        return psf_cube

    def get_psf(self, optical_design):
        filename = get_pkg_data_filename(f'data/psf_{optical_design.name}_{self.name}.nc',
                                         package='mocksipipeline.instrument.optics')
        return xarray.open_dataarray(filename)


class SlotAperture(AbstractAperture):
    """
    Slot MOXSI aperture

    Parameters
    ----------
    diameter: `~astropy.units.Quantity`
    center_to_center_distance: `~astropy.units.Quantity`
    """

    @u.quantity_input
    def __init__(self, diameter: u.cm, center_to_center_distance: u.cm):
        self.diameter = diameter
        self.center_to_center_distance = center_to_center_distance

    @property
    def name(self):
        return f"slot_{self.center_to_center_distance.to_value('um'):.0f}"

    @property
    @u.quantity_input
    def area(self) -> u.cm**2:
        return np.pi*(self.diameter/2)**2 + self.diameter*self.center_to_center_distance

    @u.quantity_input(resolution=u.micron)
    def mask(self, resolution=1*u.micron):
        """
        Create a boolean mask for the aperture relevant to this channel.

        Parameters
        ----------
        oversample: `~astropy.units.Quantity`
            How much to oversample the resulting mask by

        Returns
        -------
        : `~xarray.DataArray`
            Aperture mask with appropriate coordinates
        """
        # NOTE: Add 5% safety factor for margin on array size relative to aperture size
        aperture_size = (self.center_to_center_distance + self.diameter) * 1.05
        # NOTE: This is done this way because np.arange annoyingly does not play well
        # with quantities
        start = -aperture_size/2
        stop = aperture_size/2
        n_step = int(np.ceil(((stop - start) / resolution).to_value(u.dimensionless_unscaled)))
        x = np.linspace(start, stop, n_step, endpoint=True)
        x, y = np.meshgrid(x, x)

        # Lower part of slot
        r = np.sqrt((x - self.center_to_center_distance/2)**2 + y**2)
        pinhole_lower = np.where(r < self.diameter/2, 0, 1)

        # Upper part of slot
        r = np.sqrt((x + self.center_to_center_distance/2)**2 + y**2)
        pinhole_upper = np.where(r < self.diameter/2, 0, 1)

        # Middle rectangle
        x_cut = np.where(
            np.logical_and(-self.center_to_center_distance/2<=x, x<=self.center_to_center_distance/2), 1, 0)
        y_cut = np.where(np.logical_and(-self.diameter/2<=y, y<=self.diameter/2), 1, 0)
        rectangle = x_cut * y_cut
        rectangle = 0 ** rectangle

        mask = pinhole_lower * pinhole_upper * rectangle
        x_coord = xarray.DataArray(data=x.value,
                                   dims=['x','y'],
                                   attrs={'unit': x.unit.to_string(format='fits')})
        y_coord = xarray.DataArray(data=y.value,
                                   dims=['x','y'],
                                   attrs={'unit': y.unit.to_string(format='fits')})
        return xarray.DataArray(data=mask,
                                dims=["x", "y"],
                                coords={'x': x_coord, 'y': y_coord})

    def __repr__(self):
        return f"""Slot Aperture
-----------------------------
name: {self.name}
diameter: {self.diameter}
center to center distance: {self.center_to_center_distance}
area: {self.area}
"""


class CircularAperture(AbstractAperture):
    """
    Circular MOXSI aperture

    Parameters
    ----------
    diameter: `~astropy.units.Quantity`
    """

    @u.quantity_input
    def __init__(self, diameter: u.cm):
        self.diameter = diameter

    @property
    def name(self):
        return f"circular_{self.diameter.to_value('um'):.0f}"

    @property
    @u.quantity_input
    def area(self) -> u.cm**2:
        return np.pi*(self.diameter/2)**2

    def mask(self, resolution=1*u.micron) -> xarray.DataArray:
        # NOTE: Add 5% safety factor for margin on array size relative to aperture size
        aperture_size = self.diameter * 1.05
        # NOTE: This is done this way because np.arange annoyingly does not play well
        # with quantities
        start = -aperture_size/2
        stop = aperture_size/2
        n_step = int(np.ceil(((stop - start) / resolution).to_value(u.dimensionless_unscaled)))
        x = np.linspace(start, stop, n_step, endpoint=True)
        x, y = np.meshgrid(x, x)
        mask = np.where(np.sqrt(x**2 + y**2)<self.diameter/2, 0, 1)
        x_coord = xarray.DataArray(data=x.value,
                                   dims=['x','y'],
                                   attrs={'unit': x.unit.to_string(format='fits')})
        y_coord = xarray.DataArray(data=y.value,
                                   dims=['x','y'],
                                   attrs={'unit': y.unit.to_string(format='fits')})
        return xarray.DataArray(data=mask,
                                dims=["x", "y"],
                                coords={'x': x_coord, 'y': y_coord})

    def __repr__(self):
        return f"""Circular Aperture
-------------------------
name: {self.name}
diameter: {self.diameter}
area: {self.area}
"""
