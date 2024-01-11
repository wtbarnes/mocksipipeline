"""
Optical design instances for different MOXSI layouts
"""
import astropy.units as u

from .design import OpticalDesign

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
