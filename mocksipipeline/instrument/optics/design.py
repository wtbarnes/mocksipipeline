"""
Instrument design configurations
"""
import dataclasses

import astropy.units as u
import numpy as np

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
    camera_gain: u.Quantity[u.DN / u.electron] = 1.8 * u.DN / u.electron

    def __eq__(self, value):
        if not isinstance(value, OpticalDesign):
            raise TypeError(f'Cannot compare equality with type {type(value)}')
        dict_1 = dataclasses.asdict(self)
        dict_2 = dataclasses.asdict(value)
        return dict_1 == dict_2

    @property
    @u.quantity_input
    def spatial_plate_scale(self) -> u.Unit('arcsec / pix'):
        pixel_size = u.Quantity([self.pixel_size_x, self.pixel_size_y]) / u.pixel
        return (pixel_size / self.focal_length).decompose() * u.radian

    @property
    @u.quantity_input
    def pixel_solid_angle(self) -> u.Unit('steradian / pix'):
        """
        This is the solid angle per pixel
        """
        area = (self.spatial_plate_scale[0] * u.pix) * (self.spatial_plate_scale[1] * u.pix)
        return area / u.pix

    @property
    @u.quantity_input
    def spectral_plate_scale(self) -> u.Unit('Angstrom / pix'):
        r"""
        The spectral plate scale is computed as,

        .. math::

            \Delta\lambda = \frac{d(\Delta x\|\cos{\gamma}\| + \Delta y\|\sin{\gamma}\|)}{f^\prime}

        where :math:`\gamma` is the grating roll angle and :math`\Delta x,\Delta y`
        are the spatial plate scales, :math:`d` is the groove spacing of the grating, and
        :math:`f^\prime` is the distance between the grating and the detector.
        """
        # NOTE: Purposefully not dividing by the spectral order here as this is
        # meant to only represent the first order spectral plate scale due to how we
        # express the wavelength axis in the WCS as a "dummy" third axis. The additional dispersion
        # at orders > 1 is handled by the spectral order term in the PC_ij matrix.
        eff_pix_size = (self.pixel_size_x * np.fabs(np.cos(self.grating_roll_angle)) +
                        self.pixel_size_y * np.fabs(np.sin(self.grating_roll_angle))) / u.pix
        return self.grating_groove_spacing * (eff_pix_size / self.grating_focal_length).decompose()
