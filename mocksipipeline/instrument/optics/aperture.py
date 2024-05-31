"""
Classes for configuring MOXSI apertures
"""
import abc

import astropy.units as u
import numpy as np
import xarray

__all__ = ['SlotAperture', 'CircularAperture']


class AbstractAperture(abc.ABC):
    """
    An abstract base class for defining the geometry of a MOXSI aperture
    """

    @property
    @abc.abstractmethod
    @u.quantity_input
    def area(self) -> u.cm ** 2:
        ...

    @property
    @abc.abstractmethod
    def mask(self):
        ...


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
    @u.quantity_input
    def area(self) -> u.cm**2:
        return np.pi*(self.diameter/2)**2 + self.diameter*self.center_to_center_distance

    def mask(self, oversample=8/u.micron):
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
        n_x = int((aperture_size * oversample).to_value(u.dimensionless_unscaled))
        x = np.linspace(-aperture_size/2, aperture_size/2, n_x, endpoint=True)
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
    @u.quantity_input
    def area(self) -> u.cm**2:
        return np.pi*(self.diameter/2)**2

    def mask(self, oversample=8/u.micron) -> xarray.DataArray:
        # NOTE: Add 5% safety factor for margin on array size relative to aperture size
        aperture_size = self.diameter * 1.05
        n_x = int((aperture_size * oversample).to_value(u.dimensionless_unscaled))
        x = np.linspace(-aperture_size/2, aperture_size/2, n_x, endpoint=True)
        x, y = np.meshgrid(x, x)
        mask = np.where(np.sqrt(x**2 + y**2)<self.diameter/2, 0, 1)
        return xarray.DataArray(
            data=mask,
            dims=["x", "y"],
            coords={'x': (["x", "y"], x), 'y': (["x", "y"], y)}
        )

    def __repr__(self):
        return f"""Circular Aperture
-------------------------
diameter: {self.diameter}
area: {self.area}
"""
