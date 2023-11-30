"""
Module for modeling counts at the detector plane
"""
import copy

import astropy.units as u
import dask.array
import ndcube
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.wcs.utils import pixel_to_pixel, wcs_to_celestial_frame
from ndcube.extra_coords import QuantityTableCoordinate
from overlappy.util import strided_array
from scipy.interpolate import interp1d
from synthesizAR.instruments.util import extend_celestial_wcs

from mocksipipeline.detector.response import get_all_dispersed_channels

__all__ = [
    'convolve_with_response',
    'project_spectral_cube',
    'compute_flux_point_source',
]


def convolve_with_response(cube, channel, electrons=True, include_gain=False):
    """
    Convolve spectral cube with wavelength response to convert spectra to instrument units.

    Parameters
    ----------
    cube : `ndcube.NDCube`
    channel : `Channel`
    electrons : `bool`, optional
        If True, include conversion factor from photons to electrons.
    include_gain : `bool`, optional
        If True, include conversion fractor from electrons to DN. Cannot be true
        if `electrons` is False.

    Return
    ------
    : `ndcube.NDCube`
        Spectral cube in detector units convolved with instrument response
    """
    # Compute instrument response
    response = channel.effective_area
    if electrons:
        response *= channel.electron_per_photon
        if include_gain:
            response *= channel.camera_gain
    # Multiply by the spatial plate scale (factor of sr)
    response *= channel.pixel_solid_angle
    # NOTE: multiplying by the spacing of the wavelength array as this is
    # not generally the same as the spectral plate scale.
    response *= np.gradient(channel.wavelength)

    # Interpolate spectral cube to the wavelength array of the channel
    # FIXME: In cases where we are using a binned spectra,
    # we should be *rebinning* the spectral cube, not just interpolating it.
    # The spectra should be rebinned to the range and bin width of the
    # instrument.
    # NOTE: it is ok that our spectral cube is not necessarily guaranteed
    # to be at the spectral plate scale of the instrument as we will reproject
    # to the correct spectral plate scale at a later point in the pipeline
    # NOTE: this is only ok if we are reprojecting the full cube to the full
    # overlappogram in one go. With the per slice approach, the wavelength
    # grids must be the same, i.e. the wavelength grid exposed by the channel
    # will be the wavelength grid used by overlappogram in the wavelength axis
    cube_wavelength = cube.axis_world_coords(0)[0].to_value('Angstrom', equivalencies=u.spectral())
    f_response = interp1d(cube_wavelength,
                          cube.data,
                          axis=0,
                          bounds_error=False,
                          fill_value=0.0,)  # Response is 0 outside of the response range
    data_interp = f_response(channel.wavelength.to_value('Angstrom'))
    data_interp = (data_interp.T * response.to_value()).T

    unit = cube.unit * response.unit
    meta = copy.deepcopy(cube.meta)
    meta['CHANNAME'] = channel.name
    # Reset the units if they were in the metadata
    meta.pop('BUNIT', None)

    # Construct new WCS for the modified wavelength axis
    # NOTE: When there is only one axis that corresponds to wavlength, then
    # just dconstruct a new wavelength axis
    if len(cube.data.shape) == 1:
        new_wcs = QuantityTableCoordinate(channel.wavelength,
                                          names='wavelength',
                                          physical_types='em.wl').wcs
    else:
        new_wcs = extend_celestial_wcs(cube[0].wcs.low_level_wcs,
                                       channel.wavelength,
                                       'wavelength',
                                       'em.wl')

    return ndcube.NDCube(data_interp, wcs=new_wcs, meta=meta, unit=unit)


def project_spectral_cube(spec_cube,
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
    n_rows = channel.detector_shape[0]
    n_cols = channel.detector_shape[1]
    # NOTE: pixel-to-pixel mapping cannot handle mapping an empty list so have to special
    # case when we have only zeros in our sample; this can happen in 1 s exposures, particularly
    # at the higher spectral orders.
    if all([idx.size == 0 for idx in idx_nonzero]):
        idx_nonzero_overlap = [[], []]
    else:
        idx_nonzero_overlap = pixel_to_pixel(instr_cube.wcs, overlap_wcs, *idx_nonzero[::-1])
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
            flux = _blur_spectra(flux, blur, channel)
        cube_list.append((f'order_{channel.spectral_order}', flux))

    flux_total = ndcube.NDCube(flux_total, wcs=cube_list[0][1].wcs, unit=flux.unit)
    if blur:
        flux_total = _blur_spectra(flux_total, blur, channels[0])
    cube_list.append(('total', flux_total))

    return ndcube.NDCollection(cube_list)


def _blur_spectra(flux, blur, channel):
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
