"""
Instrument design configurations
"""
import dataclasses

import astropy.units as u

__all__ = [
    'OpticalDesign',
]


@dataclasses.dataclass
class OpticalDesign:
    r"""
    A class representing a MOXSI design configuration

    Parameters
    ----------
    name: `str`
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
    camera_gain: `~astropy.units.Quantity`, optional
        Gain of the camera that determines conversion between electrons
        and DN in the detector.
    """
    name: str
    focal_length: u.Quantity[u.cm]
    grating_focal_length: u.Quantity[u.cm]
    grating_groove_spacing: u.Quantity[u.mm]
    grating_roll_angle: u.Quantity[u.degree]
    pixel_size_x: u.Quantity[u.micron] = 7 * u.micron
    pixel_size_y: u.Quantity[u.micron] = 7 * u.micron
    detector_shape: tuple = (1504, 2000)
    camera_gain: u.Quantity[u.ct / u.electron] = 1.8 * u.ct / u.electron

    def __repr__(self):
        return f"""MOXSI Optical Design {self.name}
=========================================
focal length: {self.focal_length}

Grating
-------
focal length: {self.grating_focal_length}
groove spacing: {self.grating_groove_spacing}
roll angle {self.grating_roll_angle}

Detector
--------
pixel size: x={self.pixel_size_x}, y={self.pixel_size_y}
shape: {self.detector_shape}
camera gain: {self.camera_gain}
"""
