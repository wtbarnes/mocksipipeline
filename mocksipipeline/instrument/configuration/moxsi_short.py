"""
Configurations for different instrument designs, including different filter combinations,
pinhole locations, and number of components.
"""
import numpy as np
import astropy.units as u

from mocksipipeline.instrument.design import InstrumentDesign
from mocksipipeline.instrument.optics.aperture import CircularAperture, SlotAperture
from mocksipipeline.instrument.optics.configuration import short_design
from mocksipipeline.instrument.optics.filter import ThinFilmFilter
from mocksipipeline.instrument.optics.response import Channel
from mocksipipeline.instrument.optics.design import OpticalDesign

__all__ = [
    'moxsi_short',
    'moxsi_short_filtergrams',
    'moxsi_short_spectrogram',
    'moxsi_short_slot'
]

# Build needed filters
table = 'Chantler'
be_thin = ThinFilmFilter('Be', thickness=8 * u.micron, xrt_table=table)
be_med = ThinFilmFilter('Be', thickness=30 * u.micron, xrt_table=table)
be_thick = ThinFilmFilter('Be', thickness=350 * u.micron, xrt_table=table)
polymide = ThinFilmFilter(elements=['C', 'H', 'N', 'O'],
                          quantities=[22, 10, 2, 5],
                          density=1.43 * u.g / u.cm ** 3,
                          thickness=1 * u.micron,
                          xrt_table=table)
al_thin = ThinFilmFilter(elements='Al', thickness=142.5 * u.nm, xrt_table=table)
al_oxide = ThinFilmFilter(elements=['Al', 'O'], quantities=[2, 3], thickness=7.5 * u.nm, xrt_table=table)

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
p_y = (short_design.detector_shape[0] / 2 - 1) / 2 + short_design.detector_shape[0] / 2
# NOTE: the order here is Cartesian, not (row, column)
# NOTE: this is 1-indexed

# Set up circular aperture
pinhole = CircularAperture(44*u.micron)

# Set up filtergrams
filtergram_1 = Channel(name='filtergram_1',
                       order=0,
                       filters=be_thin,
                       reference_pixel=(p_x, p_y, 0) * u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_2 = Channel(name='filtergram_2',
                       order=0,
                       filters=be_med,
                       reference_pixel=(p_x + window, p_y, 0) * u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_3 = Channel(name='filtergram_3',
                       order=0,
                       filters=be_thick,
                       reference_pixel=(p_x + 2 * window, p_y, 0) * u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_4 = Channel(name='filtergram_4',
                       order=0,
                       filters=[al_oxide, al_thin, polymide],
                       reference_pixel=(p_x + 3 * window, p_y, 0) * u.pix,
                       design=short_design,
                       aperture=pinhole)

# Set up spectrograms
orders = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
spectrogram_1_refpix = ((short_design.detector_shape[1] - 1) / 2,
                        (short_design.detector_shape[0] / 2 - 1) / 2,
                        0) * u.pix
spectrograms = []
for order in orders:
    chan = Channel(name='spectrogram_1',
                   order=order,
                   filters=[al_thin, al_oxide],
                   reference_pixel=spectrogram_1_refpix,
                   design=short_design,
                   aperture=pinhole)
    spectrograms.append(chan)

# Build instrument configurations
moxsi_short_filtergrams = InstrumentDesign(
    'short-filtergrams',
    [filtergram_1, filtergram_2, filtergram_3, filtergram_4]
)
moxsi_short_spectrogram = InstrumentDesign('short-dispersed', spectrograms)
moxsi_short = moxsi_short_filtergrams + moxsi_short_spectrogram

##########################################################################
# Configuration for short MOXSI with an additional dispersed slot channel
detector_center = np.array(short_design.detector_shape)[::-1] // 2

# values pulled from mechanical drawing 1/11/24
filtergram_y = 3.51 * u.mm
filtergram_x = (np.arange(4) * 3.32 - 1.5 * 3.32) * u.mm
slot_y = 3.43 * u.mm

filtergram_ref_pix = [tuple(np.array([x / short_design.pixel_size_x,
                                      filtergram_y / short_design.pixel_size_y]) + detector_center) for x in
                      filtergram_x]

dispersed_ref_pix = tuple(detector_center)

slot_ref_pix = tuple(detector_center - [0, slot_y / short_design.pixel_size_y])

filtergram_1 = Channel(name='filtergram_1',
                       order=0,
                       filters=be_thin,
                       reference_pixel=(filtergram_ref_pix[0]+(0,))*u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_2 = Channel(name='filtergram_2',
                       order=0,
                       filters=be_med,
                       reference_pixel=(filtergram_ref_pix[1]+(0,))*u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_3 = Channel(name='filtergram_3',
                       order=0,
                       filters=be_thick,
                       reference_pixel=(filtergram_ref_pix[2]+(0,))*u.pix,
                       design=short_design,
                       aperture=pinhole)
filtergram_4 = Channel(name='filtergram_4',
                       order=0,
                       filters=[al_oxide, al_thin, polymide],
                       reference_pixel=(filtergram_ref_pix[3]+(0,))*u.pix,
                       design=short_design,
                       aperture=pinhole)

# Set up spectrograms

spectrograms = []
for order in orders:
    chan = Channel(name='spectrogram_1',
                   order=order,
                   filters=[al_thin, al_oxide],
                   reference_pixel=(dispersed_ref_pix+(0,))*u.pix,
                   design=short_design,
                   aperture=pinhole)
    slot = Channel(name='spectrogram_slot',
                   order=order,
                   filters=[al_thin, al_oxide],
                   reference_pixel=(slot_ref_pix+(0,))*u.pix,
                   design=short_design,
                   aperture=SlotAperture(diameter=44 * u.micron,
                                         center_to_center_distance=440 * u.micron)
                   )
    spectrograms.append(chan)
    spectrograms.append(slot)

# Build instrument configuration
moxsi_short_slot = InstrumentDesign(
    name='slotty_moxsi',
    channel_list=[filtergram_1, filtergram_2, filtergram_3, filtergram_4] + spectrograms)
