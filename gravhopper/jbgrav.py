# python driver for the _jbgrav C module

"""Functions for calculating the N-body force from a set of particles. Calls the
C extension functions to do the calculation.
"""


from . import _jbgrav
import numpy as np
from astropy import units as u, constants as const

__all__=['direct_summation', 'direct_summation_position', 'tree_force', 'tree_force_position']

def direct_summation(snap, eps, calc_force=True, calc_potential=False):
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
    calc_force : boolean
        True if the force should be calculated. Can be set to False to only calculate
        the potential. (default: True)
    calc_potential: boolean
        True if the potential should be calculated. (default: False)
        
    Returns
    -------
    acceleration_potential : array or tuple
        Return type depends on the values of calc_force and calc_potential.

        If calc_force=True then the accelerations are returned as an (Np,3) numpy
        array of the acceleration vector calculated for each particle.

        If calc_potential=True then the potential energies are returned as an Np
        element numpy array of the potential energies calculated for each particle.

        If both are set, then the return is a tuple (accelerations, potentials,).

        If neither is set, the function does no work and returns None.
    """

    # Quit if nothing to be done
    if not (calc_force or calc_potential):
        return None
    
    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    unit_energy = const.G * unit_mass / unit_length
    desired_accel_unit = u.km / u.s / u.Myr
    desired_energy_unit = u.km**2 / u.s**2

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    jbgrav_output = _jbgrav.direct_summation(posarray, massarray, eps_in_units, calc_force, calc_potential)

    # Figure out which pieces are which to adjust units
    if calc_force:
        if calc_potential:
            # Tuple of (acceleration, potential)
            output_accel = jbgrav_output[0] * unit_accel.to(desired_accel_unit)
            output_pot = jbgrav_output[1] * unit_energy.to(desired_energy_unit)
            output = (output_accel, output_pot)
        else:
            # acceleration
            output = jbgrav_output * unit_accel.to(desired_accel_unit)
    else:
        # Only potential. calc_potential must be true because we already quit at the top of the function otherwise
        output = jbgrav_output * unit_energy.to(desired_energy_unit)
        
    return output



def tree_force(snap, eps, theta=0.7, calc_force=True, calc_potential=False):
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
    calc_force : boolean
        True if the force should be calculated. Can be set to False to only calculate
        the potential. (default: True)
    calc_potential: boolean
        True if the potential should be calculated. (default: False)

    Returns
    -------
    acceleration_potential : array or tuple
        Return type depends on the values of calc_force and calc_potential.

        If calc_force=True then the accelerations are returned as an (Np,3) numpy
        array of the acceleration vector calculated for each particle.

        If calc_potential=True then the potential energies are returned as an Np
        element numpy array of the potential energies calculated for each particle.

        If both are set, then the return is a tuple (accelerations, potentials,).

        If neither is set, the function does no work and returns None.
    """
    
    # Quit if nothing to be done
    if not (calc_force or calc_potential):
        return None

    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    unit_energy = const.G * unit_mass / unit_length
    desired_accel_unit = u.km / u.s / u.Myr
    desired_energy_unit = u.km**2 / u.s**2

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    jbgrav_output = _jbgrav.tree_force(posarray, massarray, eps_in_units, theta, calc_force, calc_potential)

    # Figure out which pieces are which to adjust units
    if calc_force:
        if calc_potential:
            # Tuple of (acceleration, potential)
            output_accel = jbgrav_output[0] * unit_accel.to(desired_accel_unit)
            output_pot = jbgrav_output[1] * unit_energy.to(desired_energy_unit)
            output = (output_accel, output_pot)
        else:
            # acceleration
            output = jbgrav_output * unit_accel.to(desired_accel_unit)
    else:
        # Only potential. calc_potential must be true because we already quit at the top of the function otherwise
        output = jbgrav_output * unit_energy.to(desired_energy_unit)
        
    return output


def direct_summation_position(snap, force_pos, eps, calc_force=True, calc_potential=False):
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
    calc_force : boolean
        True if the force should be calculated. Can be set to False to only calculate
        the potential. (default: True)
    calc_potential: boolean
        True if the potential should be calculated. (default: False)
        
    Returns
    -------
    acceleration_potential : array or tuple
        Return type depends on the values of calc_force and calc_potential.

        If calc_force=True then the accelerations are returned as an (N,3) numpy
        array of the acceleration vector calculated for each position.

        If calc_potential=True then the potential energies are returned as an N
        element numpy array of the potential energies calculated for each position.

        If both are set, then the return is a tuple (accelerations, potentials,).

        If neither is set, the function does no work and returns None.
    """

    # Quit if nothing to be done
    if not (calc_force or calc_potential):
        return None
    
    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    unit_energy = const.G * unit_mass / unit_length
    desired_accel_unit = u.km / u.s / u.Myr
    desired_energy_unit = u.km**2 / u.s**2

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    eps_in_units = eps.to(unit_length).value
    forceposarray = force_pos.to(unit_length).value
    jbgrav_output = _jbgrav.direct_summation_position(posarray, massarray, forceposarray, eps_in_units, calc_force, calc_potential)

    # Figure out which pieces are which to adjust units
    if calc_force:
        if calc_potential:
            # Tuple of (acceleration, potential)
            output_accel = jbgrav_output[0] * unit_accel.to(desired_accel_unit)
            output_pot = jbgrav_output[1] * unit_energy.to(desired_energy_unit)
            output = (output_accel, output_pot)
        else:
            # acceleration
            output = jbgrav_output * unit_accel.to(desired_accel_unit)
    else:
        # Only potential. calc_potential must be true because we already quit at the top of the function otherwise
        output = jbgrav_output * unit_energy.to(desired_energy_unit)
        
    return output


def tree_force_position(snap, force_pos, eps, theta=0.7, calc_force=True, calc_potential=False):
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
    calc_force : boolean
        True if the force should be calculated. Can be set to False to only calculate
        the potential. (default: True)
    calc_potential: boolean
        True if the potential should be calculated. (default: False)

    Returns
    -------
    acceleration_potential : array or tuple
        Return type depends on the values of calc_force and calc_potential.

        If calc_force=True then the accelerations are returned as an (N,3) numpy
        array of the acceleration vector calculated for each position.

        If calc_potential=True then the potential energies are returned as an N
        element numpy array of the potential energies calculated for each position.

        If both are set, then the return is a tuple (accelerations, potentials,).

        If neither is set, the function does no work and returns None.
    """

    # Quit if nothing to be done
    if not (calc_force or calc_potential):
        return None

    positions = snap['pos']
    masses = snap['mass']

    # create the unitless numpy arrays we need. Oh natural units how I loathe thee.
    unit_length = u.kpc
    unit_mass = u.Msun
    unit_accel = const.G * unit_mass / (unit_length**2)
    unit_energy = const.G * unit_mass / unit_length
    desired_accel_unit = u.km / u.s / u.Myr
    desired_energy_unit = u.km**2 / u.s**2

    posarray = positions.to(unit_length).value
    massarray = masses.to(unit_mass).value
    forceposarray = force_pos.to(unit_length).value
    eps_in_units = eps.to(unit_length).value
    jbgrav_output = _jbgrav.tree_force_position(posarray, massarray, forceposarray, eps_in_units, theta, calc_force, calc_potential)

    # Figure out which pieces are which to adjust units
    if calc_force:
        if calc_potential:
            # Tuple of (acceleration, potential)
            output_accel = jbgrav_output[0] * unit_accel.to(desired_accel_unit)
            output_pot = jbgrav_output[1] * unit_energy.to(desired_energy_unit)
            output = (output_accel, output_pot)
        else:
            # acceleration
            output = jbgrav_output * unit_accel.to(desired_accel_unit)
    else:
        # Only potential. calc_potential must be true because we already quit at the top of the function otherwise
        output = jbgrav_output * unit_energy.to(desired_energy_unit)
        
    return output
