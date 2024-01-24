"""
Script to prep level 1 files
"""
import pathlib

import aiapy.calibrate
import asdf
import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter
from sunpy.coordinates import Helioprojective

import mocksipipeline.instrument.configuration


def apply_instrument_corrections(smap, pointing_table, correction_table):
    # Apply any needed L1 corrections
    if 'AIA' in smap.instrument:
        smap = aiapy.calibrate.update_pointing(smap, pointing_table=pointing_table)
        smap = aiapy.calibrate.correct_degradation(smap,
                                                   correction_table=correction_table,
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


def build_new_header(smap, new_frame, new_scale: u.arcsec/u.pix):
    # NOTE: by default, the reference pixel will be set such that this coordinate
    # corresponds to the center of the image.
    ref_coord = SkyCoord(0, 0, unit='arcsec', frame=new_frame)
    # Construct a new shape that encompasses the full disk
    extent = 2 * sunpy.map.solar_angular_radius(ref_coord) * 1.25
    new_shape = tuple(np.ceil((extent / new_scale[::-1]).to_value('pix')).astype(int))
    new_header = sunpy.map.make_fitswcs_header(
        new_shape,
        ref_coord,
        scale=new_scale,
        rotation_angle=0*u.deg,
        telescope=smap.meta.get('telescop', None),
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


def reproject_map(smap, new_header):
    # Explicitly using adaptive reprojection here as using interpolation when resampling to
    # very different resolutions (e.g. AIA to XRT) can lead to large differences
    # as compared to the original image.
    # NOTE: Explicitly conserving flux here. The corresponding response functions used to
    # perform the DEM inversion are multiplied by the appropriate plate scale to account for
    # the fact that many pixels are being effectively summed together.
    with Helioprojective.assume_spherical_screen(smap.observer_coordinate, only_off_disk=True):
        _smap = smap.reproject_to(astropy.wcs.WCS(new_header),
                                  algorithm='adaptive',
                                  conserve_flux=True,
                                  boundary_mode='strict',
                                  kernel='Gaussian')
    # NOTE: we manually rebuild the Map in order to preserve the metadata and to also fill in
    # the missing values
    new_data = _smap.data
    new_data[np.isnan(new_data)] = np.nanmin(new_data)
    return sunpy.map.Map(new_data, new_header)


def convolve_with_psf(smap, psf_width):
    # PSF width is specified in order (x-like, y-like) but
    # gaussian_filter expects array index ordering
    w = psf_width.to_value('pixel')[::-1]
    return smap._new_instance(gaussian_filter(smap.data, w), smap.meta)


if __name__ == '__main__':
    # Load the instrument design in order to get the appropriate scale to reproject to
    instrument_design = getattr(mocksipipeline.instrument.configuration, snakemake.config['instrument_design'])
    # Replace negative values with zeros
    m = sunpy.map.Map(snakemake.input[0])
    m = m._new_instance(np.where(m.data < 0, 0, m.data), m.meta)
    # Apply any needed corrections prior to reprojection
    pointing_table = astropy.table.QTable.read(snakemake.input[2])
    correction_table = astropy.table.QTable.read(snakemake.input[3])
    m = apply_instrument_corrections(m, pointing_table, correction_table)
    # Reproject map to common WCS
    with asdf.open(snakemake.input[1]) as af:
        ref_frame = af.tree['frame']
    new_header = build_new_header(m, ref_frame, instrument_design.optical_design.spatial_plate_scale)
    m = reproject_map(m, new_header)
    # Save map
    output_dir = pathlib.Path(snakemake.output[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    m.save(snakemake.output[0])
