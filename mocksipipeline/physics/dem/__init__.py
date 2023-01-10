"""
Skeleton for DEM Models
"""
import parse
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
import aiapy.response
import astropy.time
import ndcube
import xrtpy

from sunkit_dem import Model

from .dem_models import *  # this registers the relevant models


class DemModel:

    def __init__(self, temperature_bin_edges, collection=None, spectral_table=None, dem_model='hk12',
                 include_cross_calibration=True):
        self.temperature_bin_edges = temperature_bin_edges
        self.collection = collection
        self.spectral_table = spectral_table
        self.dem_model = dem_model
        self.include_cross_calibration = include_cross_calibration

    @property
    def temperature_bin_centers(self):
        # NOTE: should get temperature bin centers directly from the spectra that we integrate over
        # to get the temperature responses
        # Or do we integrate everything?
        logt = np.log10(self.temperature_bin_edges.to_value('K'))
        return 10**((logt[1:] + logt[:-1])/2) * u.K

    @property
    def spectral_table(self):
        return self._spectral_table

    @spectral_table.setter
    def spectral_table(self, value):
        from mocksipipeline.physics.spectral import SpectralModel
        self._spectral_table = SpectralModel(spectral_table=value).spectral_table

    @property
    def celestial_wcs(self):
        # NOTE: This is extracted from the original collection (rather than the 
        # resulting DEM) because the DEM has a gwcs that makes it currently
        # impossible to extract the celestial FITS WCS from. Once that is fixed,
        # this will not be necessary.
        # NOTE: It does not matter which key is selected. They all have the same
        # WCS.
        key_0 = list(self.collection.keys())[0]
        return self.collection[key_0].wcs

    def get_cross_calibration_factor(self, key):
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
        if not self.include_cross_calibration:
            return 1
        if 'Be' in key:
            return 1.5
        else:
            return 2.5

    @property
    def response_kernels(self):
        kernels = {}
        for key in self.collection:
            # TODO: make these tests a bit more strict
            is_aia = 'angstrom' in key.lower()
            is_xrt = 'open' in key.lower()
            if is_aia:
                _key = u.Quantity(*key.split())
                c = aiapy.response.Channel(_key)
                # NOTE: Intentionally not including the obstime here to include the degradation correction
                # because the input maps have already been corrected for degradation.
                response = c.wavelength_response() * c.plate_scale
                wavelength = c.wavelength
            elif is_xrt:
                # NOTE: The filter wheel designations can be in either order
                _key = parse.parse('{filter}-open', key.lower()) or parse.parse('open-{filter}', key.lower())
                _key = '-'.join(_key['filter'].split())
                # NOTE: Intentionally setting the date to near the start of the XRT mission to effectively have
                # zero contamination. The current implementation of the contamination data in xrtpy throws an 
                # exception for dates outside of the range of the contamination data and this data does not extend
                # over the entire mission.
                # NOTE: This date is being chosen because there is a serious bug in xrtpy that results in this being
                # the only date that does not return all NaN values for the effective area. Until that bug is resolved,
                # this date must be hardcoded to exactly this value. See https://github.com/HinodeXRT/xrtpy/issues/97.
                date = astropy.time.Time("2006-09-22T22:45:45")
                trf = xrtpy.response.TemperatureResponseFundamental(_key, date)
                ea = trf.effective_area()
                gain = const.c * const.h / trf.channel_wavelength / trf.ev_per_electron / trf.ccd_gain_right
                # NOTE: the xrtpy package makes use of the new unit in astropy DN which is much more commonly used
                # in solar physics. However, DN is not a unit recognized in the FITS standard so we substitute it 
                # for count. This is currently the unit that we use in sunpy in place of DN.
                gain = gain * u.count / u.DN
                response = ea * gain * trf.solid_angle_per_pixel
                response *= self.get_cross_calibration_factor(key)
                # NOTE: This is somewhat confusingly in units of ph Angstroms
                wavelength = trf.channel_wavelength.to_value('ph Angstrom') * u.angstrom
            else:
                raise KeyError(f'Unrecognized key {key}. Should be an AIA channel or XRT filter wheel combination.')
            T, tresp = compute_temperature_response(self.spectral_table, wavelength, response, return_temperature=True)
            kernels[key] = np.interp(self.temperature_bin_centers, T, tresp)
        
        return kernels

    @property
    def dem_settings(self):
        """
        Keyword arguments used for running the DEM inversion.
        These will change depending on the kind of model being run.
        """
        return {
            # NOTE: some experimentation seems to show that this set of parameters
            # yields reasonably smooth solutions.
            'hk12': {
                'alpha': 2.0,
                'increase_alpha': 1.5,
                'max_iterations': 50,  
            }
        }

    def run(self):
        # Run the DEM model and return a DEM data cube with dimensions space, space, temperature
        dem_model = Model(
            self.collection,
            self.response_kernels,
            self.temperature_bin_edges,
            model=self.dem_model,
        )
        return dem_model.fit(**self.dem_settings[self.dem_model])


def compute_temperature_response(spectra,
                                 instrument_wavelength,
                                 instrument_response,
                                 return_temperature=False):
    """
    Generate a temperature response from a spectra and a wavelength response.

    Parameters
    ----------
    spectra: `~ndcube.NDCube`
        Isothermal spectra as a function of temperature and wavelength.
    instrument_wavelength: `~astropy.units.Quantity`
    instrument_response: `~astropy.units.Quantity`
    """
    # Interpolate response to spectra
    spectra_wavelength = spectra.axis_world_coords(1)[0].to('Angstrom')
    f_response = interp1d(instrument_wavelength.to_value(spectra_wavelength.unit),
                          instrument_response.to_value(),
                          axis=0,
                          bounds_error=False,
                          fill_value=0.0,)
    spectra_response = u.Quantity(f_response(spectra_wavelength.value), instrument_response.unit)
    # Integrate over wavelength
    delta_wave = np.diff(spectra_wavelength)[0]  # It is assumed that this is uniform
    spectra_data = u.Quantity(spectra.data, spectra.unit)
    temperature_response = (spectra_data * spectra_response * delta_wave).sum(axis=1)
    if return_temperature:
        temperature = spectra.axis_world_coords(0)[0].to('K')
        return temperature, temperature_response
    else:
        return temperature_response
