"""
Script to compute DEM cube constrained by EUV and SXR images.
"""
import pathlib

import aiapy.response
import astropy.units as u
import ndcube
import numpy as np
import parse
import sunpy.map
import xrtpy
from astropy.nddata import StdDevUncertainty
from sunkit_dem import GenericModel, Model

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
            # NOTE: This is somewhat confusingly in units of ph Angstroms
            wavelength = trf.channel_wavelength.to_value('ph Angstrom') * u.angstrom
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


class HK12Model(GenericModel):

    def _model(self, alpha=1.0, increase_alpha=1.5, max_iterations=10, guess=None, use_em_loci=False, **kwargs):
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        from demregpy import dn2dem
        dem, edem, elogt, chisq, dn_reg = dn2dem(
            self.data_matrix.T,
            errors,
            self.kernel_matrix.T,
            np.log10(self.kernel_temperatures.to_value(u.K)),
            self.temperature_bin_edges.to_value(u.K),
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
            **kwargs,
        )
        _key = self._keys[0]
        dem_unit = self.data[_key].unit / self.kernel[_key].unit / self.temperature_bin_edges.unit
        uncertainty = edem.T * dem_unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit
        T_error_upper = self.temperature_bin_centers * (10**elogt - 1 )
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10**elogt)
        return {'dem': dem,
                'uncertainty': uncertainty,
                'em': em,
                'temperature_errors_upper': T_error_upper.T,
                'temperature_errors_lower': T_error_lower.T,
                'chi_squared': np.atleast_1d(chisq).T}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'hk12'


def compute_em(collection, kernels, temperature_bin_edges, kernel_temperatures):
    # Run the DEM model and return a DEM data cube with dimensions space, space, temperature
    dem_settings = {
        'alpha': 1.0,
        'increase_alpha': 1.5,
        'max_iterations': 50,
        'use_em_loci': True,
        'emd_int': True,
        'l_emd': True,
    }
    dem_model = Model(collection,
                      kernels,
                      temperature_bin_edges,
                      kernel_temperatures=kernel_temperatures,
                      model='hk12')
    dem_res = dem_model.fit(**dem_settings)
    # NOTE: not clear why there are negative values when resulting DEM
    # should be strictly positive
    dem_data = np.where(dem_res['em'].data<0.0, 0.0, dem_res['em'].data)
    return ndcube.NDCube(dem_data,
                         wcs=dem_res['em'].wcs,
                         meta=dem_res['em'].meta,
                         unit=dem_res['em'].unit,
                         mask=dem_res['em'].mask)


if __name__ == '__main__':
    # Build collection
    all_maps = sunpy.map.Map(snakemake.input)
    collection = build_map_collection(all_maps)
    # Construct temperature bins
    delta_log_t = float(snakemake.config['delta_log_t'])
    temperature_bin_edges = 10**np.arange(
        float(snakemake.config['log_t_left_edge']),
        float(snakemake.config['log_t_right_edge'])+delta_log_t,
        delta_log_t,
    )*u.K
    # Read in spectral table
    spectral_table_name = snakemake.config['spectral_table']
    if pathlib.Path(spectral_table_name).is_file():
        from synthesizAR.atomic.idl import read_spectral_table
        spectral_table = read_spectral_table(spectral_table_name)
    else:
        spectral_table = get_spectral_tables()[spectral_table_name]
    # Compute temperature response functions
    temperature_kernel = 10**np.arange(5, 8, 0.05)*u.K
    kernels = calculate_response_kernels(collection,
                                         temperature_kernel,
                                         spectral_table)
    # Compute EM cube
    em_cube = compute_em(collection,
                         kernels,
                         temperature_bin_edges,
                         temperature_kernel)
    # Save to disk
    write_cube_with_xarray(em_cube, 'temperature', all_maps[0].wcs, snakemake.output[0])
