"""Converts between Astropy-style and Pynbody-style units, which are not
directly compatible."""

from astropy import units as u, constants as const
from fractions import Fraction
import pynbody.units as pynu

__all__=['pynbody_to_astropy','astropy_to_pynbody']

# Direct mapping of astropy units for all pynbody units
unitmapper_pyn2ap = {
    'm':u.m,
    's':u.s,
    'kg':u.kg,
    'K':u.K,
    'rad':u.radian,
    'yr':u.yr,
    'kyr':u.kyr,
    'Myr':u.Myr,
    'Gyr':u.Gyr,
    'Hz':u.Hz,
    'kHz':u.kHz,
    'MHz':u.MHz,
    'GHz':u.GHz,
    'THz':u.THz,
    'angst':u.AA,
    'cm':u.cm,
    'mm':u.mm,
    'nm':u.nm,
    'km':u.km,
    'au':u.au,
    'pc':u.pc,
    'kpc':u.kpc,
    'Mpc':u.Mpc,
    'Gpc':u.Gpc,
    'sr':u.sr,
    'deg':u.deg,
    'arcmin':u.arcmin,
    'arcsec':u.arcsec,
    'Msol':u.Msun,
    'g':u.g,
    'm_p':const.m_p,
    'm_e':const.m_e,
    'N':u.N,
    'dyn':u.dyn,
    'J':u.J,
    'erg':u.erg,
    'eV':u.eV,
    'keV':u.keV,
    'MeV':u.MeV,
    'W':u.W,
    'Jy':u.Jy,
    'Pa':u.Pa,
    'k':const.k_B,
    'c':const.c,
    'G':const.G,
    'hP':const.h
}
# Invert mapping from astropy units to pynbody base equivalents
unitmapper_ap2pyn = {}
for k, v in unitmapper_pyn2ap.items():
    # If it's a unit, use the Astropy string repr, otherwise use the Pynbody symbol too.
    if isinstance(v, u.UnitBase):
        unitmapper_ap2pyn[str(v)] = k
    else:
        unitmapper_ap2pyn[k] = k
    

def pynbody_to_astropy(pynunit):
    """Return an astropy unit equivalent to the pynbody unit input.
    
    Parameters
    ----------
    pynunit : pynbody.units.Unit
        Pynbody unit to convert
        
    Returns
    -------
    apunit : astropy.unit.Unit
        Equivalent astropy unit. If the input unit is a composite, the function will attempt
        to compose it in the same way, but sometimes that is not possible; however, it
        will always be equivalent numerically.
        
    Raises
    ------
    pynbody.UnitsException
        If it cannot successfully convert the unit
    """        
#    See also
#    --------
#    `astropy_to_pynbody` : Converts astropy units to pynbody units.
#    :meth:`~gravhopper.IC.from_pyn_snap` : Converts a pynbody SimSnap to Gravhopper ICs
#    """
        
    pynunit_str = str(pynunit)
    
    # Does it have a direct astropy equivalent?
    if pynunit_str in unitmapper_pyn2ap:
        return unitmapper_pyn2ap[pynunit_str]
        
    # Must be a composite.
    if not isinstance(pynunit, pynu.CompositeUnit):
        raise pynu.UnitsException(f"Pynbody unit {pynunit_str} has no astropy equivalent and is not composite")
        
    # Decompose it
    astropy_equivalent = pynunit._scale
    for base, power in zip(pynunit._bases, pynunit._powers):
        # Each base had better have an astropy equivalent
        base_str = str(base)
        if base_str in unitmapper_pyn2ap:
            astropy_base = unitmapper_pyn2ap[base_str]
        else:
            raise pynu.UnitsException(f"Pynbody unit {base_str} has no astropy equivalent")
        
        astropy_equivalent *= astropy_base**power
        
    return u.Unit(astropy_equivalent)


def astropy_to_pynbody(apunit):
    """Return a pynbody unit equivalent to the astropy unit input.
    
    Parameters
    ----------
    apunit : astropy.unit.Unit
        Astropy unit to convert
        
    Returns
    -------
    pynunit : pynbody.units.Unit
        Equivalent Pynbody unit. If the input unit is a composite, the function will attempt
        to compose it in the same way, but sometimes that is not possible; however, it
        will always be equivalent numerically.
        
    Raises
    ------
    pynbody.UnitsException
        If it cannot successfully convert the unit
    """
#     See also
#     --------
#     `pynbody_to_astropy` : Converts astropy units to pynbody units.
#     :meth:`~gravhopper.Simulation.pyn_snap` : Converts a Simulation snapshot to a Pynbody SimSnap
#     """
    
    # Emergency base units for each dimension.
    defaults = (u.kpc, u.s, u.Msun, u.K, u.radian, u.sr)
        
    # Does it have a direct pynbody equivalent?
    apunit_str = str(apunit)
    if apunit_str in unitmapper_ap2pyn:
        return pynu.Unit(unitmapper_ap2pyn[apunit_str])
    
    # If it's a quantity, split into scale and unit
    if isinstance(apunit, u.Quantity):
        scale = apunit.value
        decomp_unit = apunit.unit
    elif isinstance(apunit, (u.Unit, u.CompositeUnit)):
        scale = apunit.scale
        decomp_unit = apunit
    else:
        raise pynu.UnitsException(f"Error decomposing Astropy unit {apunit}")
    
    # Construct a CompositeUnit
    pynbody_equivalent = pynu.CompositeUnit(scale, [], [])
    
    # Go through each decomposed piece
    for base, power in zip(decomp_unit.bases, decomp_unit.powers):
        # See if each base has a pynbody equivalent
        base_str = str(base)
        if base_str in unitmapper_ap2pyn:
            compose_pynbody_unit(pynbody_equivalent, 1, [base], [power])
        elif isinstance(base, u.PrefixUnit):
            # If it's a prefix unit, decompose it and try again
            prefix_decomp = base.decompose()
            compose_pynbody_unit(pynbody_equivalent, prefix_decomp.scale, prefix_decomp.bases, prefix_decomp.powers)        
        else:
            # Decompose into emergency bases
            default_decomp = base.decompose(defaults)
            compose_pynbody_unit(pynbody_equivalent, default_decomp.scale, default_decomp.bases, default_decomp.powers)
                                
    return pynbody_equivalent


def compose_pynbody_unit(unit_so_far, scale, bases, powers):
    """Add bases and powers and scale to existing pynbody composite unit."""
    
    unit_so_far._scale *= scale
    for base, power in zip(bases, powers):
        # See if each base has a pynbody equivalent
        base_str = str(base)
        if base_str in unitmapper_ap2pyn:
            pyn_base = pynu.Unit(unitmapper_ap2pyn[base_str])
            unit_so_far._bases.append(pyn_base)
            if isinstance(power, float):
                fracpower = Fraction(power)
            else:
                fracpower = power
            unit_so_far._powers.append(fracpower)
        else:
            raise pynu.UnitsException(f"Error decomposing Astropy unit {base_str}")
        
