"""
Module for projecting emission to detector plane
"""
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame
from overlappy.reproject import reproject_to_overlappogram

from mocksipipeline.detector.response import convolve_with_response

__all__ = ['DetectorComponent']


class DetectorComponent:

    @u.quantity_input
    def __init__(self, channel, roll_angle=-90*u.deg, dispersion_angle=0*u.deg):
        self.channel = channel
        self.roll_angle = roll_angle
        self.dispersion_angle = dispersion_angle

    def compute(self, spectral_cube, include_gain=True):
        instr_cube = convolve_with_response(spectral_cube, self.channel, include_gain=include_gain)
        return reproject_to_overlappogram(
            instr_cube,
            self.channel.detector_shape,
            observer=wcs_to_celestial_frame(spectral_cube.wcs).observer,
            reference_pixel=self.channel.reference_pixel,
            reference_coord=(
                0*u.arcsec,
                0*u.arcsec,
                instr_cube.axis_world_coords(0)[0].to('angstrom')[0],
            ),
            scale=(
                self.channel.resolution[0],
                self.channel.resolution[1],
                self.channel.spectral_resolution,
            ),
            roll_angle=self.roll_angle,
            dispersion_angle=self.dispersion_angle,
            order=self.channel.spectral_order,
            meta_keys=['CHANNAME'],
            use_dask=True,
            sum_over_lambda=True,
            algorithm='interpolation',
        )