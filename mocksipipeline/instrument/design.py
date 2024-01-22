"""
Class for holding full instrument design
"""
import dataclasses
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import astropy.units as u

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
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
