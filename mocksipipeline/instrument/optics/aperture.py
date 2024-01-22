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
    def area(self) -> u.cm ** 2:
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
        return np.pi * (self.diameter/2) ** 2 + self.diameter * self.center_to_center_distance

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

    def __repr__(self):
        return f"""Circular Aperture
-------------------------
diameter: {self.diameter}
area: {self.area}
"""
