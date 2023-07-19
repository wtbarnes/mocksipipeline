"""
Module for projecting emission to detector plane
"""
import astropy.units as u
from astropy.wcs.utils import wcs_to_celestial_frame, pixel_to_pixel
import dask.array
import ndcube
import numpy as np
from overlappy.reproject import reproject_to_overlappogram
from overlappy.util import strided_array

from mocksipipeline.detector.response import convolve_with_response, SpectrogramChannel, Channel

__all__ = [
    'DetectorComponent',
    'DispersedComponent',
    'FiltergramComponent',
    'sample_and_remap_spectral_cube',
]


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

    def __init__(self, channel_kwargs=None, **kwargs):
        if channel_kwargs is None:
            channel_kwargs = {}
        components = []
        for order in self.spectral_orders:
            channel = SpectrogramChannel(order, **channel_kwargs)
            component = DetectorComponent(channel, **kwargs)
            components.append(component)
        self.components = components

    @property
    def spectral_orders(self):
        return [-4, -3, -2, -1, 0, 1, 2, 3, 4]

    def compute(self, spectral_cube, **kwargs):
        results = {}
        for component in self.components:
            results[component.channel.spectral_order] = component.compute(spectral_cube, **kwargs)
        return results


class FiltergramComponent:

    def __init__(self, channel_kwargs=None, **kwargs):
        if channel_kwargs is None:
            channel_kwargs = {}
        components = []
        for name in self.filtergram_names:
            channel = Channel(name, **channel_kwargs)
            component = DetectorComponent(channel, **kwargs)
            components.append(component)
        self.components = components

    @property
    def filtergram_names(self):
        return [f'filtergram_{i}' for i in range(1, 5)]

    def compute(self, spectral_cube, **kwargs):
        results = {}
        for component in self.components:
            results[component.channel.name] = component.compute(spectral_cube, **kwargs)
        return results


def sample_and_remap_spectral_cube(spec_cube, channel, dt=1*u.s, interval=20*u.s, chunks=None, **wcs_kwargs):
    """
    Map spectral cube to detector plane via Poisson distribution

    Parameters
    ----------
    spec_cube
    channel
    dt
    interval
    """
    # Convert to instrument units
    observer = wcs_to_celestial_frame(spec_cube.wcs).observer
    instr_cube = convolve_with_response(spec_cube, channel, electrons=False, include_gain=False)
    # Sample distribution
    lam = (instr_cube.data * instr_cube.unit * u.pix * dt).to_value('photon')
    if chunks is None:
        chunks = lam.shape
    lam = dask.array.from_array(lam, chunks=chunks)
    num_iterations = int(np.ceil((interval / dt).decompose()))
    samples = dask.array.random.poisson(lam=lam, size=(num_iterations,)+lam.shape).sum(axis=0)
    samples = samples.compute()
    idx_nonzero = np.where(samples > 0)
    weights = samples[idx_nonzero]
    # Map counts to detector coordinates
    overlap_wcs = channel.get_wcs(observer, **wcs_kwargs)
    idx_nonzero_overlap = pixel_to_pixel(instr_cube.wcs, overlap_wcs, *idx_nonzero[::-1])
    n_rows = channel.detector_shape[0]
    n_cols = channel.detector_shape[1]
    hist, _, _ = np.histogram2d(idx_nonzero_overlap[1], idx_nonzero_overlap[0],
                                bins=(n_rows, n_cols),
                                range=([-.5, n_rows-.5], [-.5, n_cols-.5]),
                                weights=weights)
    return ndcube.NDCube(strided_array(hist, channel.wavelength.shape[0]),
                         wcs=overlap_wcs,
                         unit='photon')
