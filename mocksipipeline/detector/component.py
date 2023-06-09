"""
Module for projecting emission to detector plane
"""
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame
from overlappy.reproject import reproject_to_overlappogram

from mocksipipeline.detector.response import convolve_with_response, SpectrogramChannel

__all__ = ['DetectorComponent', 'DispersedComponent', 'FiltergramComponent']


class DetectorComponent:

    @u.quantity_input
    def __init__(self, channel, roll_angle=-90*u.deg, dispersion_angle=0*u.deg):
        self.channel = channel
        self.roll_angle = roll_angle
        self.dispersion_angle = dispersion_angle

    def compute(self, spectral_cube, include_gain=False, electrons=True):
        instr_cube = convolve_with_response(spectral_cube,
                                            self.channel,
                                            include_gain=include_gain,
                                            electrons=electrons)
        return reproject_to_overlappogram(
            instr_cube,
            self.channel.detector_shape,
            observer=wcs_to_celestial_frame(spectral_cube.wcs).observer,
            reference_pixel=self.channel.reference_pixel,
            reference_coord=(
                0*u.arcsec,
                0*u.arcsec,
                0*u.angstrom,
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


class DispersedComponent:

    def __init__(self, filter, channel_kwargs=None, **kwargs):
        if channel_kwargs is None:
            channel_kwargs = {}
        components = []
        for order in self.spectral_orders:
            channel = SpectrogramChannel(order, filter, **channel_kwargs)
            component = DetectorComponent(channel, **kwargs)
            components.append(component)
        self.components = components

    @property
    def spectral_orders(self):
        return [-3, -2, -1, 0, 1, 2, 3]

    def compute(self, spectral_cube, **kwargs):
        results = {}
        for component in self.components:
            results[component.channel.spectral_order] = component.compute(spectral_cube, **kwargs)
        return results


class FiltergramComponent:
    ...
