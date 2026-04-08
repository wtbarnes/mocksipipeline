"""
Script to compute DEM cube constrained by EUV and SXR images.
"""
import pathlib
import warnings

import aiapy.response
import astropy.units as u
import ndcube
import numpy as np
import parse
import sunpy.map
import xrtpy
from astropy.nddata import StdDevUncertainty
from sunkit_dem import GenericModel, Model
from simple_reg_dem import simple_reg_dem
from mocksipipeline.spectral import (compute_temperature_response,
                                     get_spectral_tables)
from mocksipipeline.util import write_cube_with_xarray


def build_map_collection(map_list):
    intensity_maps = {}
    error_maps = {}
    for m in map_list:
        key = str(m.measurement)
        if m.meta.get('measrmnt') == 'uncertainty':
            error_maps[key] = StdDevUncertainty(m.quantity)
        else:
            intensity_maps[key] = m
    return ndcube.NDCollection(
        [(k, ndcube.NDCube(m.quantity,
                           wcs=m.wcs,
                           meta=m.meta,
                           uncertainty=error_maps[k]))
         for k, m in intensity_maps.items()],
        aligned_axes=(0, 1),
    )


def get_cross_calibration_factor(key):
    """
    Factor to multiply XRT response functions by

    This is needed to resolve excess emission in XRT relative to other instruments.
    Per discussions with P.S. Athiray, best to use 1.5 for Be channels and 2.5 for
    all other channels. Also see the following papers for a more full discussion of
    these cross-calibration factors:

    - Schmelz et al. (2015) https://doi.org/10.1088/0004-637X/806/2/232
    - Schmelz et al. (2016) https://iopscience.iop.org/article/10.3847/1538-4357/833/2/182
    - Wright et al. (2017) https://iopscience.iop.org/article/10.3847/1538-4357/aa7a59
    - Athiray et al. (2020) https://doi.org/10.3847/1538-4357/ab7200
    """
    if 'Be' in key:
        return 1.5
    else:
        return 2.5


def calculate_response_kernels(collection, temperature, spectral_table):
    kernels = {}
    for key in collection:
        # NOTE: Make a map here to make it easier to access the needed
        # properties
        smap = sunpy.map.Map(collection[key].data, collection[key].meta)
        # NOTE: Explicitly calculating the plate scale here as the the maps in the
        # collection likely do not have the nominal plate scale and this is needed
        # to compute the temperature response function.
        # NOTE: We multiply by pixel because the plate scale should be in units of
        # arcsecond^2 per pixel and each scale factor of the map has units of
        # arcsecond per pixel.
        pix_solid_angle = smap.scale.axis1 * smap.scale.axis2 * u.pix
        if 'AIA' in smap.instrument:
            c = aiapy.response.Channel(smap.wavelength)
            # NOTE: Intentionally not including the obstime here to include the degradation correction
            # because the input maps have already been corrected for degradation.
            response = c.wavelength_response() * pix_solid_angle
            wavelength = c.wavelength
        elif 'XRT' in smap.instrument:
            # NOTE: The filter wheel designations can be in either order
            _key = parse.parse('{filter}-open', key.lower()) or parse.parse('open-{filter}', key.lower())
            _key = '-'.join(_key['filter'].split())
            trf = xrtpy.response.TemperatureResponseFundamental(_key, smap.date)
            ea = trf.effective_area()
            wavelength = trf.wavelength
            gain = wavelength.to('eV', equivalencies=u.equivalencies.spectral()) / u.photon
            gain /= (trf.ev_per_electron * trf.ccd_gain_right)
            response = ea * gain * pix_solid_angle
            response *= get_cross_calibration_factor(key)
        else:
            raise KeyError(f'Unrecognized key {key}. Should be an AIA channel or XRT filter wheel combination.')
        T, tresp = compute_temperature_response(spectral_table, wavelength, response, return_temperature=True)
        kernels[key] = np.interp(temperature, T, tresp)
        # NOTE: This explicit unit conversion is just to ensure there are no units issues when doing the inversion
        # (since units are stripped off in the actual calculation).
        kernels[key] = kernels[key].to('cm5 DN pix-1 s-1')

    return kernels


class SimpleRegModel(GenericModel):
    """
    Simple regularized DEM inversion model.

    This model uses Joe Plowman's simple_reg_dem algorithm which performs a regularized
    inversion with smoothness constraints to compute the DEM.
    """

    def _model(self, kmax=100, kcon=5, steps=[0.1, 0.5], drv_con=8.0, chi2_th=1.0, tol=0.1, **kwargs):
        """
        Parameters
        ----------
        kmax: `int`
            Maximum number of iterations (default: 100)
        kcon: `int`
            Initial number of steps before terminating if chi^2 never improves (default: 5)
        steps: array-like
            Two element list containing [small, large] step sizes (default: [0.1, 0.5])
        drv_con: `float`
            Derivative constraint - threshold limiting change in log DEM per unit log temperature (default: 8.0)
        chi2_th: `float`
            Reduced chi^2 threshold for termination (default: 1.0)
        tol: `float`
            Tolerance for convergence to chi2_th (default: 0.1)

        Returns
        -------
        : `dict`
            Dictionary containing 'dem', 'em', and 'chi_squared' arrays
        """
        # Inputs
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        exptimes = np.array([self.data[k].meta.get('exptime') for k in self._keys])
        logt = np.log10(self.kernel_temperatures.to_value(u.K))
        # Call simple_reg_dem
        dems, chi2 = simple_reg_dem(
            self.data_matrix.T,
            errors,
            exptimes,
            logt,
            self.kernel_matrix.T,
            kmax=kmax,
            kcon=kcon,
            steps=steps,
            drv_con=drv_con,
            chi2_th=chi2_th,
            tol=tol,
        )
        # Transpose so temperature is first axis
        dems = dems.T
        # Calculate units
        _key = self._keys[0]
        dem_unit = self.data[_key].unit / self.kernel[_key].unit / self.temperature_bin_edges.unit
        em_unit = self.data[_key].unit / self.kernel[_key].unit
        # Convert DEM to EM by multiplying by delta log T
        delta_log_t = np.diff(np.log10(self.temperature_bin_edges.to_value(u.K)))
        em = (dems * delta_log_t[:, np.newaxis, np.newaxis]) * em_unit
        dem = dems * dem_unit
        return {
            'dem': dem,
            'em': em,
            'chi_squared': np.atleast_1d(chi2).T
        }

    @classmethod
    def defines_model_for(cls, *args, **kwargs):
        return kwargs.get('model') == 'simple_reg_dem'


def compute_em(collection, kernels, temperature_bin_edges, kernel_temperatures, **kwargs):
    """
    Run the simple_reg_dem model and return an EM data cube.

    Parameters
    ----------
    collection: `ndcube.NDCollection`
        Collection of NDCubes with intensity data and uncertainties
    kernels: `dict`
        Dictionary of temperature response functions for each channel
    temperature_bin_edges: `~astropy.units.Quantity`
        Temperature bin edges
    kernel_temperatures: `~astropy.units.Quantity`
        Temperatures at which kernels are evaluated
    kwargs:
        Settings to pass to the simple_reg_dem algorithm

    Returns
    -------
    : `ndcube.NDCube`
        Emission measure cube with dimensions [n_temp, nx, ny]
    """
    dem_settings = {
        'kmax': 100,
        'kcon': 5,
        'steps': [0.1, 0.5],
        'drv_con': 8.0,
        'chi2_th': 1.0,
        'tol': 0.1,
    }
    dem_model = Model(
        collection,
        kernels,
        temperature_bin_edges,
        kernel_temperatures=kernel_temperatures,
        model='simple_reg_dem'
    )
    # Fit the model
    dem_res = dem_model.fit(**dem_settings)
    # Ensure non-negative values
    em_data = dem_res['em'].data
    if not np.all(em_data >= 0.0):
        warnings.warn("The EM array is not strictly positive, replacing negatives with 0.0")
        em_data = np.where(em_data < 0.0, 0.0, em_data)

    return ndcube.NDCube(
        em_data,
        wcs=dem_res['em'].wcs,
        meta=dem_res['em'].meta,
        unit=dem_res['em'].unit,
        mask=dem_res['em'].mask
    )


if __name__ == '__main__':
    # Read in the maps and correction table
    all_maps = sunpy.map.Map(snakemake.input[:-1])
    # Build collection
    collection = build_map_collection(all_maps)
    # Construct temperature bins
    delta_log_t = float(snakemake.config['delta_log_t'])
    temperature_bin_edges = 10**np.arange(
        float(snakemake.config['log_t_left_edge']),
        float(snakemake.config['log_t_right_edge']) + delta_log_t,
        delta_log_t,
    ) * u.K
    # Read in spectral table
    spectral_table_name = snakemake.config['spectral_table']
    if pathlib.Path(spectral_table_name).is_file():
        from synthesizAR.atomic.idl import read_spectral_table
        spectral_table = read_spectral_table(spectral_table_name)
    else:
        spectral_table = get_spectral_tables()[spectral_table_name]
    # Compute temperature response functions
    # Calculate temperature bin centers for kernel evaluation
    logt_edges = np.log10(temperature_bin_edges.to_value(u.K))
    logt_centers = (logt_edges[:-1] + logt_edges[1:]) / 2.0
    temperature_kernel = 10**logt_centers * u.K
    kernels = calculate_response_kernels(
        collection,
        temperature_kernel,
        spectral_table,
    )
    # Compute EM cube using simple_reg_dem
    em_cube = compute_em(
        collection,
        kernels,
        temperature_bin_edges,
        temperature_kernel
    )
    # Save to disk
    write_cube_with_xarray(em_cube, 'temperature', all_maps[0].wcs, snakemake.output[0])
