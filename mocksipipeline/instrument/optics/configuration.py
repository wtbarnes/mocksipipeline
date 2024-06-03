"""
Optical design instances for different MOXSI layouts
"""
import astropy.units as u

from .design import OpticalDesign

__all__ = ['short_design', 'long_design', 'short_as_built_design']


short_design = OpticalDesign(
    'short_tube',
    focal_length=19.5 * u.cm,
    grating_focal_length=19.5 * u.cm,
    grating_groove_spacing=1 / 5000 * u.mm,
    grating_roll_angle=0 * u.deg,
)

long_design = OpticalDesign(
    'long_tube',
    focal_length=25.5 * u.cm,
    grating_focal_length=25.5 * u.cm,
    grating_groove_spacing=1 / 5000 * u.mm,
    grating_roll_angle=0 * u.deg,
)

short_as_built_design = OpticalDesign(
    'short_tube_as-built',
    focal_length=19.5 * u.cm,
    grating_focal_length=18.21 * u.cm,
    grating_groove_spacing=1 / 5000 * u.mm,
    grating_roll_angle=0 * u.deg,
)
