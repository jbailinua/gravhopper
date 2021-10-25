# python driver for the _jbgrav C module

from . import _jbgrav
import numpy as np
from astropy import units as u, constants as const

__all__=['direct_summation', 'tree_force']

def direct_summation(snap, eps):
       """Calculate the gravitational acceleration on every particle in
       the simulation snapshot from every other particle in the snapshot
       using direct summation. Uses a C module for speed.
       snap should be a dictionary where snap['pos'] is an Np,3 array of
       astropy Quantities for the positions and snap['mass'] is an Np array
       of astropy Quantities for the masses."""
       
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



def tree_force(snap, eps):
       """Calculate the gravitational acceleration on every particle in
       the simulation snapshot from every other particle in the snapshot
       using a Barnes-Hut tree. Uses a C module for speed.
       snap should be a dictionary where snap['pos'] is an Np,3 array of
       astropy Quantities for the positions and snap['mass'] is an Np array
       of astropy Quantities for the masses."""
       
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
       forcearray = _jbgrav.tree_force(posarray, massarray, eps_in_units)

       return forcearray * unit_accel.to(desired_accel_unit)





