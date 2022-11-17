"""
DEM models and utilities
"""
import astropy.units as u
import numpy as np
from sunkit_dem import GenericModel

from .dem_algorithms import simple_reg_dem, sparse_em_init, sparse_em_solve, dn2dem_pos
try:
    from .dem_algorithms_fast import simple_reg_dem_gpu, simple_reg_dem_numba, simple_reg_dem_jax
except ImportError:
    pass

__all__ = ['HK12Model', 'PlowmanModel', 'CheungModel']


class PlowmanModel(GenericModel):

    def _model(self, **kwargs):
        # Reshape some of the data
        data_array = self.data_matrix.to_value('ct pix-1').T
        uncertainty_array = np.array([self.data[k].uncertainty.array for k in self._keys]).T
        tresp_array = self.kernel_matrix.to_value('cm^5 ct pix-1 s-1').T
        logt = self.temperature_bin_centers.to_value('K')
        # Assume exposure times
        exp_unit = u.Unit('s')
        exp_times = np.array([self.data[k].meta['exptime'] for k in self._keys])

        # Solve
        method = kwargs.pop('method', None)
        if method == 'gpu':
            dem, chi2 = simple_reg_dem_gpu(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        elif method == 'numba':
            dem, chi2 = simple_reg_dem_numba(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        elif method == 'jax':
            dem, chi2 = simple_reg_dem_jax(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        else:
            dem, chi2 = simple_reg_dem(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )

        # Reshape outputs
        data_units = self.data_matrix.unit / exp_unit
        dem_unit = data_units / self.kernel_matrix.unit / self.temperature_bin_edges.unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit

        return {'dem': dem,
                'em': em,
                'chi_squared': np.atleast_1d(chi2).T,
                'uncertainty': np.zeros(dem.shape)}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'plowman'


class CheungModel(GenericModel):

    def _model(self, init_kwargs=None, solve_kwargs=None):
        # Extract needed keyword arguments
        init_kwargs = {} if init_kwargs is None else init_kwargs
        solve_kwargs = {} if solve_kwargs is None else solve_kwargs

        # Reshape some of the data
        exp_unit = u.Unit('s')
        exp_times = np.array([self.data[k].meta['exptime'] for k in self._keys])
        tr_list = self.kernel_matrix.to_value('cm^5 ct / (pix s)')
        logt_list = len(tr_list) * [np.log10(self.temperature_bin_centers.to_value('K'))]
        data_array = self.data_matrix.to_value().T
        uncertainty_array = np.array([self.data[k].uncertainty.array for k in self._keys]).T

        # Call model initializer
        k_basis_int, _, basis_funcs, _ = sparse_em_init(logt_list, tr_list, **init_kwargs)
        # Solve
        coeffs, _, _ = sparse_em_solve(data_array,
                                       uncertainty_array,
                                       exp_times,
                                       k_basis_int,
                                       **solve_kwargs)

        # Compute product between coefficients and basis functions
        # NOTE: I am assuming that all basis functions are computed on the same temperature
        # array defined by temperature_bin_centers
        # TODO: Need to verify that the cheung method returns per deltalogT
        delta_logT = np.diff(np.log10(self.temperature_bin_edges.to_value('K')))
        em = np.tensordot(coeffs, basis_funcs, axes=(2, 0)) * delta_logT

        # Reshape outputs
        data_units = self.data_matrix.unit / exp_unit
        em = u.Quantity(em.T, data_units / self.kernel_matrix.unit)
        dem = (em.T / np.diff(self.temperature_bin_edges)).T

        return {'dem': dem,
                'em': em,
                'uncertainty': np.zeros(dem.shape)}

    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'cheung'


class HK12Model(GenericModel):

    def _model(self, alpha=1.0, increase_alpha=1.5, max_iterations=10, guess=None, use_em_loci=False, use_dask=False):
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
            self.data_matrix.value.T,
            errors,
            self.kernel_matrix.value.T,
            np.log10(self.temperature_bin_centers.to(u.K).value),
            self.temperature_bin_edges.to(u.K).value,
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
            use_dask=use_dask,
        )
        dem_unit = self.data_matrix.unit / self.kernel_matrix.unit / self.temperature_bin_edges.unit
        uncertainty = edem.T * dem_unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit
        T_error_upper = self.temperature_bin_centers * (10**elogt -1 )
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10**elogt)
        return {'dem': dem,
                'uncertainty': uncertainty,
                'em': em,
                'temperature_errors_upper': T_error_upper,
                'temperature_errors_lower': T_error_lower,
                'chi_squared': np.atleast_1d(chisq)}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'hk12'
