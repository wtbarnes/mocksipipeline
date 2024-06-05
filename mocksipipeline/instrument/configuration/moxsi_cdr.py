"""
Configurations for MOXSI as of CDR
"""
import astropy.units as u
import numpy as np

from mocksipipeline.instrument.design import InstrumentDesign
from mocksipipeline.instrument.optics.aperture import (CircularAperture,
                                                       SlotAperture)
from mocksipipeline.instrument.optics.configuration import \
    short_as_built_design
from mocksipipeline.instrument.optics.filter import ThinFilmFilter
from mocksipipeline.instrument.optics.response import (ALLOWED_SPECTRAL_ORDERS,
                                                       Channel)

__all__ = [
    'moxsi_cdr',
    'moxsi_cdr_filtergrams',
    'moxsi_cdr_spectrogram_pinhole',
    'moxsi_cdr_spectrogram_slot',
]

# Build needed filters
table = 'Chantler'
be_thin = ThinFilmFilter('Be', thickness=11.5*u.micron, xrt_table=table)
be_med = ThinFilmFilter('Be', thickness=29.8*u.micron, xrt_table=table)
be_thick = ThinFilmFilter('Be', thickness=350*u.micron, xrt_table=table)
polymide = ThinFilmFilter(elements=['C', 'H', 'N', 'O'],
                          quantities=[22, 10, 2, 5],
                          density=1.43*u.g/u.cm**3,
                          thickness=1*u.micron,
                          xrt_table=table)
al_thin = ThinFilmFilter(elements='Al', thickness=142.5*u.nm, xrt_table=table)
al_oxide = ThinFilmFilter(elements=['Al', 'O'], quantities=[2, 3], thickness=7.5*u.nm, xrt_table=table)

# Set up apertures
circular = CircularAperture(44*u.micron)
# NOTE: center-to-center distance is set such that the slot area is 10 times the pinhole area
slot = SlotAperture(diameter=circular.diameter,
                    center_to_center_distance=9*np.pi*circular.diameter/4)

# Set up reference pixels
detector_center = np.array(short_as_built_design.detector_shape)[::-1] / 2
# Values pulled from mechanical drawing 1/11/24
filtergram_y = 3.51 * u.mm
filtergram_x = (np.arange(4) * 3.32 - 1.5 * 3.32) * u.mm
slot_y = 3.43 * u.mm
filtergram_ref_pix = [
    tuple(np.array([x / short_as_built_design.pixel_size_x,
                    filtergram_y / short_as_built_design.pixel_size_y]) + detector_center)
    for x in filtergram_x
]
pinhole_ref_pix = tuple(detector_center)
slot_ref_pix = tuple(detector_center - [0, slot_y / short_as_built_design.pixel_size_y])

# Set up filtergrams
filtergram_1 = Channel(name='filtergram_1',
                       order=0,
                       filters=be_thin,
                       reference_pixel=(filtergram_ref_pix[0]+(0,))*u.pix,
                       design=short_as_built_design,
                       aperture=circular)
filtergram_2 = Channel(name='filtergram_2',
                       order=0,
                       filters=be_med,
                       reference_pixel=(filtergram_ref_pix[1]+(0,))*u.pix,
                       design=short_as_built_design,
                       aperture=circular)
filtergram_3 = Channel(name='filtergram_3',
                       order=0,
                       filters=be_thick,
                       reference_pixel=(filtergram_ref_pix[2]+(0,))*u.pix,
                       design=short_as_built_design,
                       aperture=circular)
filtergram_4 = Channel(name='filtergram_4',
                       order=0,
                       filters=[al_oxide, al_thin, polymide],
                       reference_pixel=(filtergram_ref_pix[3]+(0,))*u.pix,
                       design=short_as_built_design,
                       aperture=circular)

# Set up spectrograms
spectrograms_pinhole = []
spectrograms_slot = []
for order in ALLOWED_SPECTRAL_ORDERS:
    spectrograms_pinhole.append(
        Channel(name='spectrogram_pinhole',
                order=order,
                filters=[al_thin, al_oxide],
                reference_pixel=(pinhole_ref_pix+(0,))*u.pix,
                design=short_as_built_design,
                aperture=circular)
    )
    spectrograms_slot.append(
        Channel(name='spectrogram_slot',
                order=order,
                filters=[al_thin, al_oxide],
                reference_pixel=(slot_ref_pix+(0,))*u.pix,
                design=short_as_built_design,
                aperture=slot)
    )

# Build instrument configurations
moxsi_cdr_filtergrams = InstrumentDesign(
    'cdr-filtergrams',
    [filtergram_1, filtergram_2, filtergram_3, filtergram_4]
)
moxsi_cdr_spectrogram_pinhole = InstrumentDesign('cdr-dispersed-pinhole', spectrograms_pinhole)
moxsi_cdr_spectrogram_slot = InstrumentDesign('cdr-dispersed-slot', spectrograms_slot)
moxsi_cdr = moxsi_cdr_filtergrams + moxsi_cdr_spectrogram_pinhole + moxsi_cdr_spectrogram_slot
