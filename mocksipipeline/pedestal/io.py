"""
Convert sets of FITS files to a single Zarr dataset
"""
import astropy.io.fits
import dask.array
import numpy as np
import pandas
import tqdm
import xarray

__all__ = ['build_xarray_dataset']


def _steinharthart(coeff, scale=False):
    # scale=True is for "det0", which is the detector temperature
    def func(dn):
        r = dn / (4096 - dn)
        if scale:
            r *= 10000
        result = 1 / np.poly1d(coeff[::-1])(np.log(r)) - 273.15
        return result
    return func


# TODO: find out where these numbers come from.
_det0temp_sh = _steinharthart([1.1292E-03, 2.3411E-04, 0.0000E+00, 8.7755E-08], scale=True)
_det1temp_sh = _steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
_fpgatemp_sh = _steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
_thermadc_sh = _steinharthart([3.3540E-03, 3.0013E-04, 5.0852E-06, 2.1877E-07])


def build_xarray_dataset(fits_dir, out_dir=None, chunk_shape=None):
    """
    Build a Zarr dataset from a directory of FITS files containing CSIE images.

    This function reads in a list of FITS files containing CSIE images and builds
    an xarray dataset from them. The xarray dataset is backed by a Dask array to
    enable lazily-evaluated computation later on (e.g. for fitting).

    Parameters
    ----------
    fits_dir: path-like
        Directory containing CSIE images in FITS format. It is assumed that the
        extension of each file is ".fits" and each file with that extension is
        intended to be read.
    out_dir: path-like, optional
        Path to the Zarr dataset to create. Typically, this ends in ".zarr".
        If specified, this will save the dataset to this directory. If not
        specified, this function returns the xarray dataset.
    chunk_shape: `tuple`, optional
        Chunk shape to be used when building Dask array.
    """
    # Get all filenames
    all_fits_files = sorted(fits_dir.glob('*.fits'))
    # Build dataset
    data_shape = (2000, 1504)
    data = np.empty((len(all_fits_files),)+data_shape)
    meta_keys = [
        'DET0TEMP',
        'DET1TEMP',
        'THERMADC',
        'FPGATEMP',
        'EXPTIME',
        'DATE',
        'FRAME_ID',
    ]
    meta_arrays = {k: [] for k in meta_keys}
    for i, filename in enumerate(tqdm.tqdm(all_fits_files)):
        with astropy.io.fits.open(filename, memmap=False) as hdul:
            _header = hdul[0].header
            data[i,...] = hdul[0].data[:,:]
        for k in meta_arrays.keys():
            meta_arrays[k].append(_header[k])
    time_coord = pandas.to_datetime(meta_arrays['DATE'])
    if chunk_shape is None:
        # NOTE: Is this the most ideal chunk shape?
        chunk_shape = (None, data_shape[0]//20, data_shape[1]//15)
    data = dask.array.from_array(data, chunks=chunk_shape)
    ds = xarray.Dataset(
        {'data': (["sample", "row", "column"], data)},
        coords={
            "time": (["sample"], time_coord),
            "frame_id": (["sample"], np.array(meta_arrays['FRAME_ID'])),
            'temperature_detector_0': (["sample"], np.array(meta_arrays['DET0TEMP'])),
            'temperature_detector_1': (["sample"], np.array(meta_arrays['DET1TEMP'])),
            'temperature_adc': (["sample"], np.array(meta_arrays['THERMADC'])),
            'temperature_fpga': (["sample"], np.array(meta_arrays['FPGATEMP'])),
            'exposure_time': (["sample"], np.array(meta_arrays['EXPTIME'])),
        }
    )
    # Apply conversion functions for temperature
    ds['temperature_detector_0'].data = _det0temp_sh(ds['temperature_detector_0'])
    ds['temperature_detector_1'].data = _det1temp_sh(ds['temperature_detector_1'])
    ds['temperature_adc'].data = _thermadc_sh(ds['temperature_adc'])
    ds['temperature_fpga'].data = _fpgatemp_sh(ds['temperature_fpga'])
    # Add unit information
    ds['data'].attrs['unit'] = 'DN'
    for k in ['detector_0', 'detector_1', 'adc', 'fpga']:
        ds[f'temperature_{k}'].attrs['unit'] = 'deg C'
    ds['exposure_time'].attrs['unit'] = 'ms'
    # Write out the Zarr dataset
    if out_dir is None:
        return ds
    else:
        ds.to_zarr(out_dir, mode='w')
