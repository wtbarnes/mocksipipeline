"""
Utility functions for the pipeline
"""
import pathlib

import astropy.io.fits
import astropy.units as u
from astropy.wcs import WCS
from ndcube import NDCube
import numpy as np
import xarray

from overlappy.io import read_overlappogram
from overlappy.util import strided_array
from synthesizAR.instruments.util import extend_celestial_wcs

__all__ = [
    'read_data_cube',
    'stack_components',
    'read_cube_with_xarray',
    'write_cube_with_xarray',
]


def read_data_cube(filename, hdu=0, use_fitsio=False):
    """
    Helper function for reading in data cubes from a FITS
    file to an NDCube

    Parameters
    ----------
    filename: `str` or path-like
    hdu: `int`
    use_fitsio: `bool`
        Use `fitsio` package for reading files. Can be very
        useful for reading compressed FITS files very quickly.
    """
    if use_fitsio:
        import fitsio
        data, header = fitsio.read(filename, ext=hdu, header=True)
        header = dict(header)
        # NOTE: For some reason, the array shape in the header is
        # incorrect so need to reset it.
        for i, shape in enumerate(data.shape[::-1]):
            header[f'NAXIS{i+1}'] = shape
    else:
        with astropy.io.fits.open(filename) as hdul:
            data = hdul[hdu].data
            header = hdul[hdu].header

    header.pop('KEYCOMMENTS', None)
    wcs = WCS(header=header)
    unit = header.get('BUNIT', None)
    spec_cube = NDCube(data, wcs=wcs, meta=header, unit=unit)

    return spec_cube


def stack_components(components, wcs_index=0):
    """
    Combine multiple overlappogram components.

    Add multiple overlappogram components into a single overlappogram.
    This is most useful when combining various spectral orders for an
    overlappogram into a single overlappogram. The resulting array will
    be strided and it is assumed that all input component arrays are
    also strided.

    Parameters
    ----------
    components : `list`
        List of overlappogram components. These can be either filepaths
        or `~ndcube.NDCube` objects.
    wcs_index : `int`, optional
        Index of the component to use for the WCS. Defaults to 0.
    """
    layers = []
    for c in components:
        if isinstance(c, NDCube):
            layers.append(c)
        else:
            layers.append(read_overlappogram(pathlib.Path(c)))
    data = np.array([l.data[0] for l in layers]).sum(axis=0)
    wcs = layers[wcs_index].wcs
    data_strided = strided_array(data, layers[wcs_index].data.shape[0])
    return NDCube(data_strided,
                  wcs=wcs,
                  unit=layers[wcs_index].unit,
                  meta=layers[wcs_index].meta)


def read_cube_with_xarray(filename, axis_name, physical_type):
    """
    Read an xarray netCDF file and rebuild an NDCube

    This function reads a data cube from a netCDF file and rebuilds it
    as an NDCube. The assumption is that the attributes on the stored
    data array have the keys necessary to reconstitute a celestial FITS
    WCS and that the axis denoted by `axis_name` is the additional axis
    along which to extend that celestial WCS. This works only for 3D cubes
    where two of the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    filename: `str`, path-like
        File to read from, usually a netCDF file
    axis_name: `str`
        The addeded coordinate along which to extend the celestial WCS.
    physical_type: `str`
        The physical type of `axis_name` as denoted by the IVOA designation.
    """
    cube_xa = xarray.open_dataarray(filename)
    # Make quantity
    meta = cube_xa.attrs
    data = u.Quantity(cube_xa.data, meta.pop('unit'))
    # Build celestial WCS from attributes
    celestial_wcs = astropy.wcs.WCS(header=meta)
    # Get temperature axis
    axis_array = u.Quantity(cube_xa[axis_name].data, cube_xa[axis_name].attrs.get('unit'))
    # Build combined WCS with temperature axis
    combined_wcs = extend_celestial_wcs(celestial_wcs, axis_array, axis_name, physical_type)
    # Build NDCube
    return NDCube(data, wcs=combined_wcs, meta=meta)


def write_cube_with_xarray(cube, axis_name, celestial_wcs, filename):
    """
    Write an NDCube to a netCDF file

    This function writes an NDCube to a netCDF file by first expressing 
    it as an xarray DataArray. This works only for 3D cubes where two of
    the axes correspond to spatial, celestial axes.

    Parameters
    ----------
    cube: `ndcube.NDCube`
    axis_name: `str`
    celestial_wcs: `astropy.wcs.WCS`
    filename: `str` or path-like
    """
    wcs_keys = dict(celestial_wcs.to_header())
    axis_array = cube.axis_world_coords(axis_name)[0]
    axis_coord = xarray.Variable(axis_name,
                                 axis_array.value,
                                 attrs={'unit': axis_array.unit.to_string()})
    cube_xa = xarray.DataArray(
        cube.data,
        dims=[axis_name, 'lat', 'lon'],
        coords={
            axis_name: axis_coord,
        },
        attrs={**wcs_keys, 'unit': cube.unit.to_string()}
    )
    cube_xa.to_netcdf(filename)
