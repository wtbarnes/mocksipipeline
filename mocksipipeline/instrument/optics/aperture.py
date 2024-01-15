"""
Classes for configuring MOXSI apertures
"""
import abc

import astropy.units as u
import numpy as np

__all__ = ['SlotAperture', 'CircularAperture']


class AbstractAperture(abc.ABC):
    """
    An abstract base class for defining the geometry of a MOXSI aperture
    """

    @abc.abstractproperty
    @u.quantity_input
    def area(self) -> u.cm**2:
        ...


class SlotAperture(AbstractAperture):
    """
    Rectangular MOXSI aperture

    Parameters
    ----------
    dimensions: `~astropy.units.Quantity`
    """

    @u.quantity_input
    def __init__(self, dimensions: u.cm):
        self.dimensions = dimensions

    @property
    @u.quantity_input
    def area(self) -> u.cm**2:
        return self.dimensions[0]*self.dimensions[1]

    def __repr__(self):
        return f"""Rectangular Slot Aperture
-----------------------------
dimensions: {self.dimensions}
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
        return np.pi * (self.diameter / 2) ** 2

    def __repr__(self):
        return f"""Circular Aperture
-------------------------
diameter: {self.diameter}
area: {self.area}
"""
