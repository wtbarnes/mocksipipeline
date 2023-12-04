"""
Instrument design configurations
"""
import dataclasses

import astropy.units as u
import numpy as np

__all__ = [
    'InstrumentDesign',
    'nominal_design',
]


@dataclasses.dataclass
class InstrumentDesign:
    r"""
    A class representing a MOXSI design configuration

    Parameters
    ----------
    focal_length: `~astropy.units.Quantity`
        The distance from the pinhole to the detector
    grating_focal_length: `~astropy.units.Quantity`
        The distance from the grating to the detector
    grating_groove_spacing: `~astropy.units.Quantity`
        Spacing between grating bars
    grating_roll_angle: `~astropy.units.Quantity`
        Orientation of the grating relative to the horizontal
        axis of the detector.
    pixel_size_x: `~astropy.units.Quantity`, optional
        Physical size of the horizontal dimension of a
        detector pixel
    pixel_size_y: `~astropy.units.Quantity`, optional
        Physical size of the horizontal dimension of a
        detector pixel
    detector_shape: `tuple`, optional
        Number of pixels in the vertical and horizontal direction
        (in that order) on the detector.
    pinhole_diameter: `~astropy.units.Quantity`, optional
        Diameter of the circular pinhole
    camera_gain: `~astropy.units.Quantity`, optional
        Gain of the camera that determines conversion between electrons
        and DN in the detector.
    """
    focal_length: u.Quantity[u.cm]
    grating_focal_length: u.Quantity[u.cm]
    grating_groove_spacing: u.Quantity[u.mm]
    grating_roll_angle: u.Quantity[u.degree]
    pixel_size_x: u.Quantity[u.micron] = 7 * u.micron
    pixel_size_y: u.Quantity[u.micron] = 7 * u.micron
    detector_shape: tuple = (1500, 2000)
    pinhole_diameter: u.Quantity[u.micron] = 44 * u.micron
    camera_gain: u.Quantity[u.ct/u.electron] = 1.8 * u.ct/u.electron

    @property
    @u.quantity_input
    def pinhole_area(self) -> u.cm**2:
        "Area for a circular pinhole"
        return np.pi * (self.pinhole_diameter / 2)**2


nominal_design = InstrumentDesign(
    focal_length=19.5*u.cm,
    grating_focal_length=19.5*u.cm,
    grating_groove_spacing=1/5000*u.mm,
    grating_roll_angle=0*u.deg,
)
