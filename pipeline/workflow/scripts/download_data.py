"""
Script to download data for constraining the DEM
"""
import pathlib

import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.time import parse_time

import mocksipipeline.net  # Register XRTSynopticClient
from mocksipipeline.net import FilterWheel1, FilterWheel2

if __name__ == '__main__':
    obstime = parse_time(snakemake.config["obstime"])
    obstime_window = float(snakemake.config["obstime_window"]) * u.s
    time = a.Time(obstime-obstime_window/2, end=obstime+obstime_window/2, near=obstime)
    # Construct AIA query
    if snakemake.params.instrument == 'aia':
        wavelength = float(snakemake.params.filter) * u.Angstrom
        instr_query = a.Instrument.aia & a.Wavelength(wavelength)
    elif snakemake.params.instrument == 'xrt':
        # Construct XRT query
        # TODO: replace this with VSO query when that becomes viable again
        fw1, fw2 = snakemake.params.filter.split(':')
        fw_combo = FilterWheel1(fw1) & FilterWheel2(fw2)
        instr_query = a.Instrument.xrt & a.Source.hinode & a.Provider('MSU') & a.Level(2) & fw_combo
    else:
        raise ValueError(f'Unrecognized instrument {snakemake.params.instrument}')

    query = Fido.search(time, instr_query)
    outfile = pathlib.Path(snakemake.output[0])
    file = Fido.fetch(query, path=outfile.parent, overwrite=True)
    # Rename file
    pathlib.Path(file[0]).rename(outfile)
