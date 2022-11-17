import numpy as np
import thermo
from xrt.backends.raycing.materials import Material
import dataclasses
import astropy.units as u


@dataclasses.dataclass
class ThinFilmFilter:
    r"""
    A class representing a thin film filter for modeling instrument responses.

    Parameters
            *elements*: str or sequence of str
                Contains all the constituent elements (symbols)

            *quantities*: None or sequence of floats of length of *elements*
                Coefficients in the chemical formula. If None, the coefficients
                are all equal to 1.

            *thickness*: u.Quantity specifying material thickness

            *density*: density of the filter. If None, the density is looked up using the chemical formula

            *xrt_table*: str
            At the time of instantiation the tabulated scattering factors of
            each element are read and then interpolated at the requested **q**
            value and energy. *table* can be 'Henke' (10 eV < *E* < 30 keV)
            [Henke]_, 'Chantler' (11 eV < *E* < 405 keV) [Chantler]_ or 'BrCo'
            (30 eV < *E* < 509 keV) [BrCo]_.

            The tables of f2 factors consider only photoelectric
            cross-sections. The tabulation by Chantler can optionally have
            *total* absorption cross-sections. This option is enabled by
            *table* = 'Chantler total'.

        .. [Henke] http://henke.lbl.gov/optical_constants/asf.html
           B.L. Henke, E.M. Gullikson, and J.C. Davis, *X-ray interactions:
           photoabsorption, scattering, transmission, and reflection at
           E=50-30000 eV, Z=1-92*, Atomic Data and Nuclear Data Tables
           **54** (no.2) (1993) 181-342.

        .. [Chantler] http://physics.nist.gov/PhysRefData/FFast/Text/cover.html
           http://physics.nist.gov/PhysRefData/FFast/html/form.html
           C. T. Chantler, *Theoretical Form Factor, Attenuation, and
           Scattering Tabulation for Z = 1 - 92 from E = 1 - 10 eV to E = 0.4 -
           1.0 MeV*, J. Phys. Chem. Ref. Data **24** (1995) 71-643.

        .. [BrCo] http://www.bmsc.washington.edu/scatter/periodic-table.html
           ftp://ftpa.aps.anl.gov/pub/cross-section_codes/
           S. Brennan and P.L. Cowan, *A suite of programs for calculating
           x-ray absorption, reflection and diffraction performance for a
           variety of materials at arbitrary wavelengths*, Rev. Sci. Instrum.
           **63** (1992) 850-853.
    """
    elements: str = None
    quantities: float = None
    thickness: u.Quantity = 0 * u.nm
    mesh_ratio: u.Quantity = 100 * u.percent
    mesh_material: str = ''
    density: u.Quantity = None
    density_ratio: float = 1
    name: str = None
    xrt_table: str = 'Henke'
    incidence_angle: u.Quantity = 90 * u.deg

    @property
    def chemical_formula(self) -> str:
        if self.quantities is None:
            chemical_formula = self.elements

        else:
            chemical_formula = ''
            for element, quantity in zip(self.elements, self.quantities):
                chemical_formula += element + str(int(quantity))

        return chemical_formula

    @property
    def density_normalized(self) -> u.Quantity:
        if self.density is None:
            density_normalized = thermo.Chemical(self.chemical_formula).rho * u.kg / u.m ** 3
        else:
            density_normalized = self.density

        return density_normalized

    @property
    def xrt_material(self) -> Material:
        return Material(
            elements=self.elements,
            quantities=self.quantities,
            kind='plate',
            t=self.thickness,
            table=self.xrt_table,
            rho=self.density_ratio * self.density_normalized.to(
                u.g / u.cm ** 3).value,
        )

    def transmissivity(self, energy: u.Quantity) -> u.Quantity:
        absorption = self.xrt_material.get_absorption_coefficient(energy.to(u.eV).value) / u.cm
        transmissivity = np.exp(-absorption * self.thickness / np.sin(self.incidence_angle))
        return transmissivity
