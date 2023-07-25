"""
Functions for building the response matrix
"""
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.wcs
import numpy as np
import ndcube
import sunpy.map
from sunpy.coordinates import Helioprojective, get_earth
from scipy.interpolate import interp1d

from synthesizAR.atomic.idl import spectrum_to_cube

from mocksipipeline.detector.response import get_all_dispersed_channels

__all__ = [
    'compute_effective_spectra',
    'compute_response_matrix',
    'write_response_matrix',
    'read_response_matrix',
]


def compute_effective_spectra(spectra, channel):
    """
    Calculate the spectra weighted by the wavelength response of the channel in
    the wavelength direction.
    """
    f_interp = interp1d(spectra.axis_world_coords(1)[0].to_value('Angstrom'),
                        spectra.data,
                        axis=1,
                        bounds_error=False,
                        fill_value=0.0)  # Response is 0 outside of the response range
    spectra_interp = f_interp(channel.wavelength.to_value('Angstrom')) * spectra.unit
    response = channel.wavelength_response * channel.plate_scale * np.gradient(channel.wavelength)
    spectra_eff = spectra_interp * response
    return spectrum_to_cube(spectra_eff, channel.wavelength, spectra.axis_world_coords(0)[0])


def compute_response_matrix(spectral_table, roll_angle, extent=2500*u.arcsec):
    r"""
    Compute the MOXSI response matrix for a given spectra.

    From CHIANTI, we can produce spectral map of the radiative loss as a function of $\lambda$ and $T$,

    .. math::

        S(\lambda,T)\quad[\mathrm{ph}\,\mathrm{cm}^3\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{Å}^{-1}]

    Additionally, the instrument response is a function of $\lambda$ and $m$ (spectral order),

    .. math::

        \varepsilon(\lambda, m)\quad[\mathrm{cm}^2\,\mathrm{sr}\,\mathrm{pix}^{-1}\,\mathrm{DN}\,\mathrm{ph}^{-1}\mathrm{Å}]

    Combining these two gives us our intermediate response matrix,

    .. math::

        R^\prime(\lambda,m,T) = S(\lambda, T)\varepsilon(\lambda,m)\quad[\mathrm{DN}\,\mathrm{cm}^{5}\,\mathrm{s}^{-1}\,\mathrm{pix}^{-1}]

    We then want to understand how to include our mapping from world pixel to detector pixel.
    Ultimately, we want a 3D response matrix which depends on $p^\prime,p,T$,
    the pixel coordinate in the inverted frame, the pixel coordinate in the original detector,
    and the temperature, respectively,

    .. math::

        R(p^\prime,p,T)\quad[\mathrm{DN}\,\mathrm{cm}^{5}\,\mathrm{s}^{-1}\,\mathrm{pix}^{-1}]
    """
    # Compute the effective spectra
    dispersed_channels = get_all_dispersed_channels()
    spectra_eff = [compute_effective_spectra(spectral_table, chan) for chan in dispersed_channels]
    # Compute the WCS that we will invert into and that we will invert from
    shape = tuple(np.ceil((extent / dispersed_channels[0].resolution[::-1]).to_value('pix')).astype(int))
    # NOTE: I do not think it matters what this observer is, because ultimately this is just a
    # pixel-to-pixel transformation and thus all of the celestial transforms should factor out.
    observer = get_earth('2000-01-01 00:00:00')
    hpc_frame = Helioprojective(observer=observer, obstime=observer.obstime)
    ref_coord = SkyCoord(0, 0, unit='arcsec', frame=hpc_frame)
    header = sunpy.map.make_fitswcs_header(shape,
                                           ref_coord,
                                           scale=dispersed_channels[0].resolution,
                                           rotation_angle=roll_angle)
    wcs_prime = astropy.wcs.WCS(header=header)
    wcs_dispersed = [chan.get_wcs(observer, roll_angle=roll_angle) for chan in dispersed_channels]
    # Find primed pixels (sometimes called "field angles"). This is a row in the primed FOV that is
    # aligned with the dispersion direction
    # We convert to world coordinates as an intermediate step in order to perform the world-to-pixel
    # transform later.
    px_prime = np.arange(wcs_prime.array_shape[1])  # NOTE: assume dispersive axis along the x-axis
    py_prime = (wcs_prime.array_shape[0] - 1)/2
    coord_prime = wcs_prime.pixel_to_world(px_prime, py_prime)
    # Compute response matrix
    # This is 4D: primed pixel coordinates, dispersed pixel coordinates, temperature, spectral order
    # When using this, should sum over spectral order
    response_matrix_shape = (wcs_prime.array_shape[1:] +
                             wcs_dispersed[0].array_shape[2:] +
                             spectra_eff[0].data.shape[:1] +
                             (len(dispersed_channels),))
    response_matrix = np.zeros(response_matrix_shape)
    for i_order, (chan, wcs_d) in enumerate(zip(dispersed_channels, wcs_dispersed)):
        for i_wave, wave in enumerate(chan.wavelength):
            _, _, i_pix = wcs_d.world_to_array_index(coord_prime, wave)
            i_pix = np.array(i_pix)
            in_bounds = np.where(np.logical_and(i_pix >= 0, i_pix < response_matrix.shape[1]))
            # NOTE: indexing this way assumes that channel.wavelength and
            # wavelength axis of the effective spectra are aligned!
            response_matrix[px_prime[in_bounds], i_pix[in_bounds], :, i_order] += spectra_eff[i_order].data[:, i_wave]

    return response_matrix


def write_response_matrix(response_matrix, filename):
    ...


def read_response_matrix(filename):
    ...
