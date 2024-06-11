"""
Class for holding full instrument design
"""
import copy
import dataclasses
import pprint

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_earth
from sunpy.coordinates.utils import get_limb_coordinates

from mocksipipeline.instrument.optics.response import Channel


@dataclasses.dataclass
class InstrumentDesign:
    name: str
    channel_list: list[Channel]

    def plot_detector_layout(self, observer=None, wavelength=0*u.AA, colors=None, **kwargs):
        """
        Plot position of solar limb on detector for all components at a given wavelength.

        Parameters
        ----------
        observer: `~astropy.coordinates.SkyCoord`, optional
            Location of the observer used to get the limb position. Defaults to Earth.
        wavelength: `~astropy.Quantity`, optional
            Wavelength of the images to plot. This determines the degree of dispersion
        colors: `dict`, optional
            Dictionary of color options with names corresponding to
            '{channel.name}_{channel.spectral_order}'
        """
        if observer is None:
            observer = get_earth('2020-01-01')
        limb_coord = get_limb_coordinates(observer, resolution=kwargs.get('resolution', 500))
        origin = SkyCoord(Tx=0*u.arcsec, Ty=0*u.arcsec, frame='helioprojective', observer=observer)

        fig = plt.figure(figsize=kwargs.get('figsize'))
        ax = fig.add_subplot()
        for channel in self.channel_list:
            _wcs = channel.get_wcs(observer)
            px, py, _ = _wcs.world_to_pixel(limb_coord, wavelength)
            px_o, py_o, _ = _wcs.world_to_pixel(origin, wavelength)
            if colors is None:
                # Specifically call out the 0th order images
                _color=f'C{abs(channel.spectral_order)-1}' if channel.spectral_order != 0 else 'k'
            else:
                _color = colors[f'{channel.name}_{channel.spectral_order}']
            # Approximate effect of PSF as elliptical distortions of circular limb
            extent_x = (px.max() - px.min()) * self.optical_design.pixel_size_x
            extent_y = (py.max() - py.min()) * self.optical_design.pixel_size_y
            psf_extent_x = copy.copy(channel.aperture.diameter)
            psf_extent_y = copy.copy(channel.aperture.diameter)
            if hasattr(channel.aperture, 'center_to_center_distance'):
                psf_extent_y = psf_extent_y + channel.aperture.center_to_center_distance
            stretch = np.array([[(psf_extent_x + extent_x) / extent_x, 0],
                                [0, (psf_extent_y + extent_y) / extent_y]])
            delta_px, delta_py = np.dot(stretch, np.array([px-px_o, py-py_o]))
            # Plot undistorted limb
            ax.plot(px, py, ls=':', color=_color)
            # Plot distorted limb
            ax.plot(px_o + delta_px, py_o + delta_py, ls='-', color=_color)
            # Plot center position
            ax.plot(px_o, py_o, ls='', marker='x', color=_color)
            # Add labels to the zeroth order components only
            if channel.spectral_order == 0:
                if 'filtergram' in channel.name:
                    label = channel.filter_label
                else:
                    label = ' '.join(channel.name.split('_')).capitalize()
                ax.text(px_o, py_o+180, label, va='bottom', ha='center', color='k')
        ax.set_xlim(0, self.optical_design.detector_shape[1])
        ax.set_ylim(0, self.optical_design.detector_shape[0])
        ax.set_xlabel('Detector Pixel x')
        ax.set_ylabel('Detector Pixel y')
        ax.set_title(f'{wavelength=:.3f}')
        ax.set_aspect('equal')
        if kwargs.get('return_figure', False):
            return fig
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
        # There are two ways to index an Instrument configuration
        # 1. Through the string designation of the channel. This returns a single channel
        # 2. Through some slice of the channel list. This will return another instrument config
        #    or possibly a single channel
        if isinstance(value, str):
            return {f'{c.name}_{c.spectral_order}': c for c in self.channel_list}[value]
        else:
            new_channel_list = self.channel_list[value]
            if isinstance(new_channel_list, type(self.channel_list[0])):
                return new_channel_list
            else:
                return type(self)(self.name, new_channel_list)

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
