"""
Class for holding full instrument design
"""
import dataclasses
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.units as u
import pprint

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
    name: str
    channel_list: list[Channel]

    def plot_detector_layout(self):
        solar_radius = 1000 * u.arcsec  # sun with margin

        fig, ax = plt.subplots()
        for channel in self.channel_list:
            width = solar_radius / channel.spatial_plate_scale[0]
            height = solar_radius / channel.spatial_plate_scale[1]
            ax.add_patch(
                Ellipse(xy=channel.reference_pixel, width=width.value * 2, height=height.value * 2)
            )

        # this sets the plot limits to the detector shape of the last channel.  right now each channel could have a
        # different detector shape
        ax.set_xlim(0, channel.detector_shape[1])
        ax.set_ylim(0, channel.detector_shape[0])

    def __post_init__(self):
        # Each channel must have the same optical design
        if not all([c.design==self.channel_list[0].design for c in self.channel_list]):
            raise ValueError('All channels must have the same optical design.')

    def __add__(self, instrument):
        if not isinstance(instrument, InstrumentDesign):
            raise TypeError(f'Addition is not supported with types {type(instrument)}')
        combined_name = f'{self.name}+{instrument.name}'
        return InstrumentDesign(combined_name, self.channel_list+instrument.channel_list)

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
