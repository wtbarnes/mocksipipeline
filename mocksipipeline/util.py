"""
Utility functions for the pipeline
"""
import pathlib

import astropy.io.fits
from astropy.wcs import WCS
from ndcube import NDCube
import numpy as np
from overlappy.io import read_overlappogram
from overlappy.util import strided_array

__all__ = ['read_data_cube', 'stack_components']


def read_data_cube(filename, hdu=0):
    """
    Helper function for reading in data cubes from a FITS
    file to an NDCube
    """
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
