"""
Components for preparing data for DEM inversion
"""
import os

import aiapy.calibrate
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import StdDevUncertainty
import astropy.wcs
import ndcube
import numpy as np
from sunpy.coordinates import Helioprojective
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.net.attr import or_
from sunpy.time import parse_time

import mocksipipeline.net  # Register XRTSynopticClient
from mocksipipeline.net import FilterWheel1, FilterWheel2
from mocksipipeline import logger
from mocksipipeline.physics.dem import DemModel


class DataPrep(DemModel):

    def __init__(self,
                 *args,
                 map_list=None,
                 aia_correction_table=None,
                 aia_error_table=None,
                 aia_pointing_table=None,
                 use_aia_errors=False,
                 conserve_flux=True,
                 **kwargs):
        self.map_list = map_list
        self.aia_correction_table = aia_correction_table
        self.aia_error_table = aia_error_table
        self.aia_pointing_table = aia_pointing_table
        self.use_aia_errors = use_aia_errors
        self.conserve_flux = conserve_flux
        super().__init__(*args, **kwargs)

    @property
    def collection(self):
        if self._collection is None:
            logger.info('Building map collection')
            self._collection = self.build_collection()
        return self._collection

    @collection.setter
    def collection(self, value):
        self._collection = value

    def build_collection(self):
        # NOTE: it does not really matter what this is, but should be the same for all maps
        ref_frame = self.map_list[0].coordinate_frame
        cubes = []
        for m in self.map_list:
            # Replace negative values with zeros
            m = m._new_instance(np.where(m.data<0, 0, m.data), m.meta)
            # Apply any needed corrections prior to reprojection
            m = self.apply_corrections(m)
            # NOTE: need to know approximately how many pixels we combined into each new pixel
            # in order to compute the error on each map in the case of the AIA images
            n_sample = int(np.round(m.dimensions.x / self.new_shape[1] * m.dimensions.y / self.new_shape[0]).value)
            # Reproject map to common WCS
            m = self.reproject_map(m, ref_frame)
            # Compute uncertainty
            error = self.compute_uncertainty(m, n_sample)
            # Build cube
            cube = ndcube.NDCube(m.quantity, wcs=m.wcs, uncertainty=error, meta=m.meta)
            cubes.append((str(m.measurement), cube))

        map_collection = ndcube.NDCollection(cubes, aligned_axes=(0, 1))
        return map_collection

    def apply_corrections(self, smap):
        # Apply any needed L1 corrections
        if 'AIA' in smap.instrument:
            smap = aiapy.calibrate.update_pointing(smap, pointing_table=self.aia_pointing_table)
            smap = aiapy.calibrate.correct_degradation(smap,
                                                       correction_table=self.aia_correction_table,
                                                       calibration_version=8,)
            smap /= smap.exposure_time
        elif 'XRT' in smap.instrument:
            # NOTE: the level 2 maps do not have a unit designation in their header but based on
            # the description of the level 2 composite synoptic images described in Takeda et al. (2016)
            # the images have all been normalized to their exposure time and thus are in units of DN s-1
            # The reason for this is these composite images are composed of images with different
            # exposure times to account for over exposure in images where the exposure time is too short.
            smap.meta['BUNIT'] = 'DN/s'
        return smap

    @property
    def new_shape(self):
         # NOTE: calculated to include the full-disk from 1 AU
        extent = 2500 * u.arcsec
        return np.ceil((self.new_scale / extent).to_value('pix')).astype(int)

    @property
    def new_scale(self):
        return u.Quantity([6, 6], 'arcsec / pix')

    def compute_uncertainty(self, smap, n_sample):
        if 'AIA' in smap.instrument and self.use_aia_errors:
            # NOTE: If conserving flux in each pixel, this is not really doing the right
            # thing as this is the uncertainty if we were averaging the intensity in 
            # each pixel. However, if conserving flux, the intensity is being effectively
            # summed into each pixel so this uncertainty will be much too small.
            error = aiapy.calibrate.estimate_error(smap.quantity * smap.exposure_time,
                                                   smap.wavelength,
                                                   n_sample=n_sample,
                                                   error_table=self.aia_error_table)
            error /= smap.exposure_time
        else:
            # Simplest case is to just always assume %20 errors on everything
            # Adding in quadrature (statistical error) and assuming all
            # contributions to the superpixel are the mean.
            error = smap.quantity * 0.2 / np.sqrt(n_sample)
        error[np.isnan(error)] = 0.0 * error.unit
        return StdDevUncertainty(error)

    def build_new_header(self, smap, new_frame):
        # NOTE: by default, the reference pixel will be set such that this coordinate
        # corresponds to the center of the image.
        ref_coord = SkyCoord(0, 0, unit='arcsec', frame=new_frame)
        new_header = sunpy.map.make_fitswcs_header(
            self.new_shape,
            ref_coord,
            scale=self.new_scale,
            rotation_angle=0*u.deg,
            instrument=smap.instrument,
            observatory=smap.observatory,
            wavelength=smap.wavelength,
            exposure=smap.exposure_time,
            projection_code='TAN',
            unit=smap.unit / u.pix,
        )
        # NOTE: preserve the filter wheel keywords in the case of XRT. These
        # are not easily propagated through via property names on the map itself.
        if 'EC_FW1_' in smap.meta:
            new_header['EC_FW1_'] = smap.meta['EC_FW1_']
        if 'EC_FW2_' in smap.meta:
            new_header['EC_FW2_'] = smap.meta['EC_FW2_']
        return new_header

    def reproject_map(self, smap, ref_frame):
        new_header = self.build_new_header(smap, ref_frame)
        # Explicitly using adaptive reprojection here as using interpolation when resampling to
        # very different resolutions (e.g. AIA to XRT) can lead to large differences
        # as compared to the original image.
        # NOTE: Explicitly conserving flux here. The corresponding response functions used to
        # perform the DEM inversion are multiplied by the appropriate plate scale to account for
        # the fact that many pixels are being effectively summed together.
        with Helioprojective.assume_spherical_screen(smap.observer_coordinate, only_off_disk=True):
            _smap = smap.reproject_to(astropy.wcs.WCS(new_header),
                                      algorithm='adaptive',
                                      conserve_flux=self.conserve_flux,
                                      boundary_mode='strict',
                                      kernel='Gaussian')
        # NOTE: we manually rebuild the Map in order to preserve the metadata and to also fill in
        # the missing values
        new_data = _smap.data
        new_data[np.isnan(new_data)] = np.nanmin(new_data)
        return sunpy.map.Map(new_data, new_header)


class DataQuery(DataPrep):

    def __init__(self, data_directory, obstime, *args, **kwargs):
        self.data_directory = data_directory
        self.obstime = obstime
        super().__init__(*args, **kwargs)

    @property
    def map_list(self):
        if self._map_list is None:
            logger.info('Searching and downloading maps')
            self._map_list = self.build_map_list()
        return self._map_list

    @map_list.setter
    def map_list(self, value):
        self._map_list = value

    def build_map_list(self):
        file_list = self.get_data(self.obstime)
        map_list = sunpy.map.Map(file_list)
        return map_list

    @property
    def fetch_kwargs(self):
        return {
            'overwrite': True,
        }

    @property
    def aia_wavelengths(self):
        return [94, 131, 171, 193, 211, 335] * u.angstrom

    @property
    def xrt_filters(self):
        return [('Be-thin', 'Open')]

    @property
    def obstime_window(self):
        # The window around the obstime for searching
        return 1 * u.h

    def query_data(self, obstime):
        obstime = parse_time(obstime)
        time = a.Time(obstime-self.obstime_window/2,
                      end=obstime+self.obstime_window/2,
                      near=obstime)
        # Construct AIA query
        logger.debug(f'Using AIA wavelengths: {self.aia_wavelengths}')
        aia_query = a.Instrument.aia & or_(*[a.Wavelength(w) for w in self.aia_wavelengths])
        # Construct XRT query
        logger.debug(f'Using XRT filter wheel combinations: {self.xrt_filters}')
        fw_combos = [FilterWheel1(fw1) & FilterWheel2(fw2) for fw1,fw2 in self.xrt_filters]
        xrt_query = a.Instrument.xrt & a.Source.hinode & a.Provider('MSU') & a.Level(2) & or_(*fw_combos)
        query = Fido.search(time, aia_query | xrt_query)
        logger.debug(query)
        self.check_query_results(query)
        return query

    def check_query_results(self, query):
        n_results = len(query)
        n_filters = len(self.aia_wavelengths) + len(self.xrt_filters)
        if n_results != n_filters:
            logger.warn(f'Number of search results {n_results} is not equal to number of requested filters {n_filters}.')

    def download_data(self, query):
        path = os.path.join(self.data_directory, '{instrument}')
        files = Fido.fetch(query, path=path, **self.fetch_kwargs)
        # This will try again if there are errors
        Fido.fetch(files)
        # If there are still errors, 
        if files.errors:
            logger.error(files.errors)
        return files

    def get_data(self, obstime):
        query = self.query_data(obstime)
        files = self.download_data(query)
        return files
