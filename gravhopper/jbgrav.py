# python driver for the _jbgrav C module

"""Functions for calculating the N-body force from a set of particles. Calls the
C extension functions to do the calculation.
"""


from . import _jbgrav
import numpy as np
from astropy import units as u, constants as const

__all__=['direct_summation', 'direct_summation_position', 'tree_force', 'tree_force_position']

def direct_summation(snap, eps):
    """Calculate the gravitational acceleration on every particle in
    the simulation snapshot from every other particle in the snapshot
    using direct summation. Uses an external C module for speed.
       
    Parameters
    ----------
    snap : dict
        snap['pos'] must be an (Np,3) array of astropy Quantities for the positions
        of the particles, and snap['mass'] must be an (Np) array of astropy Quantities
        of the masses of each particle.
    eps : Quantity
        Gravitational softening length.
        
    Returns
    -------
    acceleration : array
        An (Np,3) numpy array of the acceleration vector calculated for each particle.
    """
    
    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    desired_accel_unit = u.km / u.s / u.Myr

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    forcearray = _jbgrav.direct_summation(posarray, massarray, eps_in_units)

    return forcearray * unit_accel.to(desired_accel_unit)



def tree_force(snap, eps, theta=0.7):
    """Calculate the gravitational acceleration on every particle in
    the simulation snapshot from every other particle in the snapshot
    using a Barnes-Hut tree. Uses an external C module for speed.
       
    Parameters
    ----------
    snap : dict
        snap['pos'] must be an (Np,3) array of astropy Quantities for the positions
        of the particles, and snap['mass'] must be an (Np) array of astropy Quantities
        of the masses of each particle.
    eps : Quantity
        Gravitational softening length.
    theta : float
        Opening angle in radians. (default: 0.7)

    Returns
    -------
    acceleration : array
        An (Np,3) numpy array of the acceleration vector calculated for each particle.
    """

    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    desired_accel_unit = u.km / u.s / u.Myr

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    forcearray = _jbgrav.tree_force(posarray, massarray, eps_in_units, theta)

    return forcearray * unit_accel.to(desired_accel_unit)


def direct_summation_position(snap, force_pos, eps):
    """Calculate the gravitational acceleration at the specified locations
    from every particle in the snapshot using direct summation.
    Uses an external C module for speed.
    
    direct_summation(snap, eps) should be equivalent to but twice as slow as
    direct_summation_position(snap, snap['pos'], eps).
       
    Parameters
    ----------
    snap : dict
        snap['pos'] must be an (Np,3) array of astropy Quantities for the positions
        of the particles, and snap['mass'] must be an (Np) array of astropy Quantities
        of the masses of each particle.
    force_pos : array
        An (N,3) array of astropy Quantities for the positions at which the force will
        be calculated. If any position exactly matches a particle position from the
        snapshot, the self-force from the particle at its own location is taken to be zero.
    eps : Quantity
        Gravitational softening length.
        
    Returns
    -------
    acceleration : array
        An (N,3) numpy array of the acceleration vector calculated for each position.
    """
    
    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    desired_accel_unit = u.km / u.s / u.Myr

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    forceposarray = force_pos.to(unit_length).value
    forcearray = _jbgrav.direct_summation_position(posarray, massarray, forceposarray, eps_in_units)

    return forcearray * unit_accel.to(desired_accel_unit)


def tree_force_position(snap, force_pos, eps, theta=0.7):
    """Calculate the gravitational acceleration at the specified locations
    from every particle in the snapshot using a Barnes-Hut tree.
    Uses an external C module for speed.
       
    Parameters
    ----------
    snap : dict
        snap['pos'] must be an (Np,3) array of astropy Quantities for the positions
        of the particles, and snap['mass'] must be an (Np) array of astropy Quantities
        of the masses of each particle.
    force_pos : array
        An (N,3) array of astropy Quantities for the positions at which the force will
        be calculated. If any position exactly matches a particle position from the
        snapshot, the self-force from the particle at its own location is taken to be zero.
    eps : Quantity
        Gravitational softening length.
    theta : float
        Opening angle in radians. (default: 0.7)

    Returns
    -------
    acceleration : array
        An (Np,3) numpy array of the acceleration vector calculated for each particle.
    """

    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    desired_accel_unit = u.km / u.s / u.Myr

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    forceposarray = force_pos.to(unit_length).value
    eps_in_units = eps.to(unit_length).value
    forcearray = _jbgrav.tree_force_position(posarray, massarray, forceposarray, eps_in_units, theta)

    return forcearray * unit_accel.to(desired_accel_unit)
