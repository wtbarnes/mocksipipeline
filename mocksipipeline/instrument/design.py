"""
Class for holding full instrument design
"""
import dataclasses
import pprint

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_earth
from sunpy.coordinates.utils import get_limb_coordinates

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
    name: str
    channel_list: list[Channel]

    def plot_detector_layout(self, observer=None, wavelength=0*u.AA, color=None):
        if observer is None:
            observer = get_earth('2020-01-01')
        limb_coord = get_limb_coordinates(observer, resolution=100)
        origin = SkyCoord(Tx=0*u.arcsec, Ty=0*u.arcsec, frame='helioprojective', observer=observer)

        fig = plt.figure()
        ax = fig.add_subplot()
        colors = {i: f'C{i}' for i in range(12)}
        for channel in self.channel_list:
            _wcs = channel.get_wcs(observer)
            px, py, _ = _wcs.world_to_pixel(limb_coord, wavelength)
            _color=colors[abs(channel.spectral_order)] if color is None else color
            ax.plot(px, py, ls='-', color=_color)
            px, py, _ = _wcs.world_to_pixel(origin, wavelength)
            ax.plot(px, py, ls='', marker='x', color=_color)
        ax.set_xlim(0, self.optical_design.detector_shape[1])
        ax.set_ylim(0, self.optical_design.detector_shape[0])
        plt.show()

    def __post_init__(self):
        # Each channel must have the same optical design
        if not all([c.design==self.channel_list[0].design for c in self.channel_list]):
            raise ValueError('All channels must have the same optical design.')

    def __add__(self, instrument):
        if not isinstance(instrument, InstrumentDesign):
            raise TypeError(f'Addition is not supported with types {type(instrument)}')
        combined_name = f'{self.name}+{instrument.name}'
        return InstrumentDesign(combined_name, self.channel_list+instrument.channel_list)

    def __getitem__(self, value):
        return {f'{c.name}_{c.spectral_order}': c for c in self.channel_list}[value]

    @property
    def optical_design(self):
        # All channels will have the same optical design
        return self.channel_list[0].design

    def __repr__(self):
        channel_reprs = '\n'.join(str(c) for c in self.channel_list)
        return f"""MOXSI Instrument Configuration {self.name}
-------------------------------------------------------------

{pprint.pformat(self.optical_design)}

Channels
========
Number of channels: {len(self.channel_list)}

{channel_reprs}
"""
