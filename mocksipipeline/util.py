"""
Utility functions for the pipeline
"""
import astropy.io.fits
from astropy.wcs import WCS
from ndcube import NDCube

__all__ = ['read_data_cube']


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
