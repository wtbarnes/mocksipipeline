"""
Module for projecting emission to detector plane
"""
import astropy.units as u
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.wcs.utils import wcs_to_celestial_frame, pixel_to_pixel
from astropy.stats import gaussian_fwhm_to_sigma
import dask.array
import ndcube
from ndcube.extra_coords import QuantityTableCoordinate
import numpy as np
from overlappy.reproject import reproject_to_overlappogram
from overlappy.util import strided_array

from mocksipipeline.detector.response import (convolve_with_response,
                                              get_all_dispersed_channels,
                                              get_all_filtergram_channels)

__all__ = [
    'DetectorComponent',
    'DispersedComponent',
    'FiltergramComponent',
    'sample_and_remap_spectral_cube',
    'dem_table_to_ndcube',
    'compute_flux_point_source',
]


class DetectorComponent:

    @u.quantity_input
    def __init__(self, channel, roll_angle=90*u.deg, dispersion_angle=0*u.deg):
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
        for channel in get_all_dispersed_channels(**channel_kwargs):
            component = DetectorComponent(channel, **kwargs)
            components.append(component)
        self.components = components

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
        for channel in get_all_filtergram_channels(**channel_kwargs):
            component = DetectorComponent(channel, **kwargs)
            components.append(component)
        self.components = components

    def compute(self, spectral_cube, **kwargs):
        results = {}
        for component in self.components:
            results[component.channel.name] = component.compute(spectral_cube, **kwargs)
        return results


def sample_and_remap_spectral_cube(spec_cube,
                                   channel,
                                   dt=1*u.s,
                                   interval=20*u.s,
                                   convert_to_dn=False,
                                   chunks=None,
                                   **wcs_kwargs):
    """
    Map spectral cube to detector plane via Poisson distribution

    Parameters
    ----------
    spec_cube
    channel
    dt
    interval
    convert_to_dn: `bool`, optional
        If True, sum counts in DN. Poisson sampling will still be done
        in photon space
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
    # NOTE: For a large number of iterations, this can cause strange behavior because of the
    # very large size of the array
    samples = dask.array.random.poisson(lam=lam, size=(num_iterations,)+lam.shape).sum(axis=0)
    samples = samples.compute()
    idx_nonzero = np.where(samples > 0)
    weights = samples[idx_nonzero]
    # (Optionally) Convert to DN
    unit = 'photon'
    if convert_to_dn:
        # NOTE: we can select the relevant conversion factors this way because the wavelength
        # axis of lam is the same as channel.wavelength and thus their indices are aligned
        ct_per_photon = channel.electron_per_photon * channel.camera_gain
        weights = weights * ct_per_photon.to_value('ct / ph')[idx_nonzero[0]]
        unit = 'ct'
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
                         unit=unit)


def compute_flux_point_source(intensity, location, blur=None, channels=None, **kwargs):
    """
    Compute flux for each channel from a point source.
    """
    electrons = kwargs.pop('electrons', False)
    include_gain = kwargs.pop('include_gain', False)
    if channels is None:
        channels = get_all_dispersed_channels()[5:]
    pix_grid, _, _ = channels[0].get_wcs(location.observer, **kwargs).world_to_pixel(location, channels[0].wavelength)
    flux_total = np.zeros(pix_grid.shape)
    cube_list = []
    for channel in channels:
        flux = convolve_with_response(intensity, channel, electrons=electrons, include_gain=include_gain)
        flux.meta['spectral_order'] = channel.spectral_order
        _pix_grid, _, _ = channel.get_wcs(location.observer, **kwargs).world_to_pixel(location, channel.wavelength)
        flux_total += np.interp(pix_grid, _pix_grid, flux.data)
        if blur:
            flux = blur_spectra(flux, blur, channel)
        cube_list.append((f'order_{channel.spectral_order}', flux))

    flux_total = ndcube.NDCube(flux_total, wcs=cube_list[0][1].wcs, unit=flux.unit)
    if blur:
        flux_total = blur_spectra(flux_total, blur, channels[0])
    cube_list.append(('total', flux_total))

    return ndcube.NDCollection(cube_list)


def blur_spectra(flux, blur, channel):
    """
    Blur spectral resolution of flux

    Parameters
    ----------
    flux:
    blur: `~astropy.units.Quantity`
        FWHM in spectral space of the estimated blur
    """
    std = blur / np.fabs(channel.spectral_order) * gaussian_fwhm_to_sigma
    std_eff = (std / channel.spectral_resolution).to_value('pix')
    kernel = Gaussian1DKernel(std_eff)
    data = convolve(flux.data, kernel)
    return ndcube.NDCube(data, wcs=flux.wcs, unit=flux.unit, meta=flux.meta)


def dem_table_to_ndcube(dem_table):
    """
    Parse an astropy table of DEM information into an NDCube

    Args:
        dem_table (_type_): _description_

    Returns:
        _type_: _description_
    """
    temperature = dem_table['temperature_bin_center']
    em = dem_table['dem']*np.gradient(temperature, edge_order=2)
    tab_coord = QuantityTableCoordinate(temperature,
                                        names='temperature',
                                        physical_types='phys.temperature')
    return ndcube.NDCube(em, wcs=tab_coord.wcs, meta=dem_table.meta)
