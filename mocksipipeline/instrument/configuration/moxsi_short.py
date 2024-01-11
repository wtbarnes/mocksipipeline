"""
Configurations for different instrument designs, including different filter combinations,
pinhole locations, and number of components.
"""
import astropy.units as u

from mocksipipeline.instrument.design import InstrumentDesign
from mocksipipeline.instrument.optics.configuration import short_design
from mocksipipeline.instrument.optics.filter import ThinFilmFilter
from mocksipipeline.instrument.optics.response import Channel

__all__ = [
    'moxsi_short',
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

# Set up filtergrams
filtergram_1 = Channel(name='filtergram_1',
                       order=0,
                       filters=be_thin,
                       reference_pixel=(p_x, p_y, 0) * u.pix,
                       design=short_design)
filtergram_2 = Channel(name='filtergram_2',
                       order=0,
                       filters=be_med,
                       reference_pixel=(p_x + window, p_y, 0) * u.pix,
                       design=short_design)
filtergram_3 = Channel(name='filtergram_3',
                       order=0,
                       filters=be_thick,
                       reference_pixel=(p_x + 2 * window, p_y, 0) * u.pix,
                       design=short_design)
filtergram_4 = Channel(name='filtergram_4',
                       order=0,
                       filters=[al_oxide, al_thin, polymide],
                       reference_pixel=(p_x + 3 * window, p_y, 0) * u.pix,
                       design=short_design)

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
                   design=short_design)
    spectrograms.append(chan)

# Build instrument configurations
moxsi_short = InstrumentDesign([filtergram_1, filtergram_2, filtergram_3, filtergram_4] + spectrograms)
