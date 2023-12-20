"""
Instrument design configurations
"""
import dataclasses
from mocksipipeline.detector.response import Channel
from mocksipipeline.detector.filter import ThinFilmFilter

import astropy.units as u
import numpy as np

__all__ = [
    'InstrumentDesign',
    'OpticalDesign',
    'nominal_design',
    'moxsi_short',
]


@dataclasses.dataclass
class OpticalDesign:
    r"""
    A class representing a MOXSI design configuration

    Parameters
    ----------
    focal_length: `~astropy.units.Quantity`
        The distance from the pinhole to the detector
    grating_focal_length: `~astropy.units.Quantity`
        The distance from the grating to the detector
    grating_groove_spacing: `~astropy.units.Quantity`
        Spacing between grating bars
    grating_roll_angle: `~astropy.units.Quantity`
        Orientation of the grating relative to the horizontal
        axis of the detector.
    pixel_size_x: `~astropy.units.Quantity`, optional
        Physical size of the horizontal dimension of a
        detector pixel
    pixel_size_y: `~astropy.units.Quantity`, optional
        Physical size of the horizontal dimension of a
        detector pixel
    detector_shape: `tuple`, optional
        Number of pixels in the vertical and horizontal direction
        (in that order) on the detector.
    pinhole_diameter: `~astropy.units.Quantity`, optional
        Diameter of the circular pinhole
    camera_gain: `~astropy.units.Quantity`, optional
        Gain of the camera that determines conversion between electrons
        and DN in the detector.
    """
    focal_length: u.Quantity[u.cm]
    grating_focal_length: u.Quantity[u.cm]
    grating_groove_spacing: u.Quantity[u.mm]
    grating_roll_angle: u.Quantity[u.degree]
    pixel_size_x: u.Quantity[u.micron] = 7 * u.micron
    pixel_size_y: u.Quantity[u.micron] = 7 * u.micron
    detector_shape: tuple = (1504, 2000)
    pinhole_diameter: u.Quantity[u.micron] = 44 * u.micron
    camera_gain: u.Quantity[u.ct / u.electron] = 1.8 * u.ct / u.electron

    @property
    @u.quantity_input
    def pinhole_area(self) -> u.cm ** 2:
        "Area for a circular pinhole"
        return np.pi * (self.pinhole_diameter / 2) ** 2


nominal_design = OpticalDesign(
    focal_length=19.5 * u.cm,
    grating_focal_length=19.5 * u.cm,
    grating_groove_spacing=1 / 5000 * u.mm,
    grating_roll_angle=0 * u.deg,
)

slot_design = OpticalDesign(
    focal_length=19.5 * u.cm,
    grating_focal_length=18.21 * u.cm,
    grating_groove_spacing=1 / 5000 * u.mm,
    grating_roll_angle=0 * u.deg,
)


@dataclasses.dataclass
class InstrumentDesign:
    channel_list: list[Channel]


# These are based off the current instrument design
table = 'Chantler'
polymide = ThinFilmFilter(elements=['C', 'H', 'N', 'O'],
                          quantities=[22, 10, 2, 5],
                          density=1.43 * u.g / u.cm ** 3,
                          thickness=1 * u.micron,
                          xrt_table=table)
aluminum_oxide = ThinFilmFilter(elements=['Al', 'O'], quantities=[2, 3], thickness=7.5 * u.nm)
aluminum = ThinFilmFilter(elements='Al', thickness=142.5 * u.nm, xrt_table=table)
aluminum_filter = [aluminum, aluminum_oxide]
be_thin = ThinFilmFilter('Be', thickness=8 * u.micron, xrt_table=table)
be_med = ThinFilmFilter('Be', thickness=30 * u.micron, xrt_table=table)
be_thick = ThinFilmFilter('Be', thickness=350 * u.micron, xrt_table=table)
al_poly = [polymide] + [aluminum_filter]
al_thin = aluminum_filter

# NOTE: this is the number of pixels between the edge of the detector
# and the leftmost and rightmost filtergram images
margin = 50
# NOTE: this is the width, in pixel space, of each filtergram image
window = 475
# NOTE: this is the x coordinate of the reference pixel of the leftmost
# filtergram image
p_x = margin + (window - 1) / 2
# NOTE: this is the y coordinate of the reference pixel of all of the
# filtergram images
p_y = (nominal_design.detector_shape[0] / 2 - 1) / 2 + nominal_design.detector_shape[0] / 2
# NOTE: the order here is Cartesian, not (row, column)
# NOTE: this is 1-indexed

filtergram_1_refpix = (p_x, p_y, 0) * u.pix,
filtergram_2_refpix = (p_x + window, p_y, 0) * u.pix
filtergram_3_refpix = (p_x + 2 * window, p_y, 0) * u.pix
filtergram_4_refpix = (p_x + 3 * window, p_y, 0) * u.pix
spectrogram_1_refpix = ((nominal_design.detector_shape[1] - 1) / 2, (nominal_design.detector_shape[0] / 2 - 1) / 2,
                        0) * u.pix

filtergram_1 = Channel(name='filtergram_1', order=0, filters=be_thin,
                       reference_pixel=filtergram_1_refpix, design=nominal_design)
filtergram_2 = Channel(name='filtergram_2', order=0, filters=be_med,
                       reference_pixel=filtergram_2_refpix, design=nominal_design)
filtergram_3 = Channel(name='filtergram_3', order=0, filters=be_thick,
                       reference_pixel=filtergram_3_refpix, design=nominal_design)
filtergram_4 = Channel(name='filtergram_4', order=0, filters=al_poly,
                       reference_pixel=filtergram_4_refpix, design=nominal_design)

orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
spectrograms = [Channel(name='spectrogram_1', order=i, filters=al_thin, reference_pixel=spectrogram_1_refpix, design=nominal_design) for i in
                orders]

moxsi_short = InstrumentDesign(
    channel_list=[filtergram_1, filtergram_2, filtergram_3, filtergram_4] + spectrograms
)

# detector_center = nominal_design.detector_shape // 2 * u.pix

# moxsi_short_slot = InstrumentDesign(
#     channel_list=[
#         Channel(name='pinhole_dispersed', order=dispersed_orders, filters=default_filters['al_thin'],
#                 reference_pixel=detector_center),
#
#     ]
# )
