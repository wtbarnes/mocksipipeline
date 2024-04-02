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

    @abc.abstractproperty
    @u.quantity_input
    def area(self) -> u.cm ** 2:
        ...

    @abc.abstractproperty
    @u.quantity_input
    def mask(self) -> xarray.DataArray:
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
    def area(self) -> u.cm ** 2:
        return np.pi * (self.diameter / 2) ** 2 + self.diameter * self.center_to_center_distance

    def mask(self, oversample=8):
        pinhole_diameter = self.diameter
        center2center_distance = self.center_to_center_distance

        aperture_size = (center2center_distance + pinhole_diameter) * 1.05
        x = np.linspace(-aperture_size / 2, aperture_size / 2, num=int((aperture_size * oversample).value),
                        endpoint=True)
        x, y = np.meshgrid(x, x)

        # shift coordinate system
        x_shift = x - center2center_distance / 2

        r = np.sqrt(x_shift * x_shift + y * y)
        pinhole = np.ones_like(r).value
        pinhole[r < pinhole_diameter / 2] = 0

        # shift coordinate system
        x_shift = x + center2center_distance / 2

        r = np.sqrt(x_shift * x_shift + y * y)
        pinhole2 = np.ones_like(r).value
        pinhole2[r < pinhole_diameter / 2] = 0

        x_cut = np.ones_like(r).value
        y_cut = np.ones_like(r).value

        x_cut *= x > -center2center_distance / 2
        x_cut *= x < center2center_distance / 2

        y_cut *= y > -pinhole_diameter / 2
        y_cut *= y < pinhole_diameter / 2

        rectangle = x_cut * y_cut
        rectangle = 0 ** rectangle

        mask = pinhole * pinhole2 * rectangle
        return xarray.DataArray(
            data=mask,
            dims=["x", "y"],
            coords=dict(
                x=(["x", "y"], x),
                y=(["x", "y"], y),
            ),
        )

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
    def area(self) -> u.cm ** 2:
        return np.pi * (self.diameter / 2) ** 2

    def mask(self, oversample=4) -> xarray.DataArray:
        big_aperture = self.diameter * 1.05
        x = np.arange(int(big_aperture.value) * oversample)
        x = x - x.shape[0] // 2

        x, y = np.meshgrid(x, x)

        r = np.sqrt(x * x + y * y)
        pinhole = np.ones_like(r)
        pinhole[r < self.diameter.value / 2 * oversample] = 0

        return xarray.DataArray(
            data=pinhole,
            dims=["x", "y"],
            coords=dict(
                x=(["x", "y"], x / oversample),
                y=(["x", "y"], y / oversample),
            ),
        )

    def __repr__(self):
        return f"""Circular Aperture
-------------------------
diameter: {self.diameter}
area: {self.area}
"""
