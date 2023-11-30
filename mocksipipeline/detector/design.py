"""
Instrument design configurations
"""
import dataclasses

import astropy.units as u
import numpy as np

__all__ = ['InstrumentDesign']


@dataclasses.dataclass
class InstrumentDesign:
    r"""
    A class representing a MOXSI design configuration

    Parameters
    ----------
    focal_length
    grating_focal_length
    grating_groove_spacing
    pixel_size_x
    pixel_size_y
    detector_shape
    grating_roll_angle
    pinhole_diameter
    """
    focal_length: u.Quantity[u.cm]
    grating_focal_length: u.Quantity[u.cm]
    grating_groove_spacing: u.Quantity[u.mm]
    pixel_size_x: u.Quantity[u.micron] = 7 * u.micron
    pixel_size_y: u.Quantity[u.micron] = 7 * u.micron
    detector_shape: tuple = (1500, 2000)
    grating_roll_angle: u.Quantity[u.degree] = 0 * u.degree
    pinhole_diameter: u.Quantity[u.micron] = 44 * u.micron
    camera_gain: u.Quantity[u.ct/u.electron] = 1.8 * u.ct/u.electron

    @property
    @u.quantity_input
    def pinhole_area(self) -> u.cm**2:
        "Area for a circular pinhole"
        return np.pi * (self.pinhole_diameter / 2)**2


# As we iterate on the design, additional configurations
# should go here

nominal_design = InstrumentDesign(
    focal_length=19.5*u.cm,
    grating_focal_length=19.5*u.cm,
    grating_groove_spacing=1/5000*u.mm,
)
