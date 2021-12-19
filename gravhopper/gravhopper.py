#!/usr/bin/py

"""
GravHopper
==========

A package for performing N-body simulations in Python, named in honor of Grace Hopper.

Written by Jeremy Bailin, University of Alabama
First initial last name at ua dot edu

Contains the following classes:

Simulation : Main class for creating and performing an N-body simulation.

IC : Static functions for generating a variety of useful initial conditions for
N-body simulations.

grav : Module containing the functions that calculate the N-body forces.

GravHopperException : Exceptions raised within GravHopper
"""


from . import jbgrav
import numpy as np
from scipy.misc import derivative
from scipy.interpolate import interp1d
from scipy import special, integrate
from astropy import units as u, constants as const
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if any of pynbody, galpy, and gala are there. If so, import them.
try:
    import pynbody as pyn
    USE_PYNBODY = True
except ImportError:
    USE_PYNBODY = False
    pass

try:
    import galpy.util
    galpy.util.__config__.set('astropy', 'astropy-units', 'True')
    import galpy.potential
    USE_GALPY = True
except ImportError:
    USE_GALPY = False
    pass

try:
    import gala.potential
    USE_GALA = True
except ImportError:
    USE_GALA = False
    pass


class GravHopperException(Exception):
    """Parent class for all error exceptions."""
    pass

class UninitializedSimulationException(GravHopperException):
    """Exception for trying to run a simulation without any initial conditions."""
    pass

class ICException(GravHopperException):
    """Exception for trying to add ICs that don't make sense."""
    def __init__(self,msg):
        print(msg)

class UnknownAlgorithmException(GravHopperException):
    """Exception for using an unknown N-body algorithm name."""
    pass
    
class ExternalPackageException(GravHopperException):
    """Exception for trying to call a pynbody/galpy/gala function when not using them."""
    def __init__(self,msg):
        print(msg)


class Simulation(object):
    """Main class for N-body simulation.
    
    Attributes
    ----------
    Np : int
        Number of particles
    Nsnap : int
        Number of snapshots
    positions : Quantity array of dimension (Nsnap,Np,3)
        numpy array of the positions of each particle at each snapshot
    velocities : Quantity array of dimension (Nsnap,Np,3)
        numpy array of the velocities of each particle at each snapshot
    masses : Quantity array of length Np
        numpy array of the masses of each particle
    times : Quantity array of length Nsnap
        numpy array of the time of each snapshot
    lenunit : Unit
        Default internal length unit
    velunit : Unit
        Default internal velocity unit
    accelunit : Unit
        Default internal acceleration unit
    """
    
    def __init__(self, dt=1*u.Myr, eps=100*u.pc, algorithm='tree'):
        """Initializes Simulation object.
        
        Parameters
        ----------
        dt : Quantity
            Time step length. Default: 1 Myr
        eps : Quantity
            Gravitational softening length. Default: 100 pc
        algorithm : str ('tree' or 'direct')
            Gravitational force calculation algorithm. Default: 'tree'

        Example
        -------
        To create a new simulation with a timestep of 1 year::
        
            sim = Simulation(dt=1.0*u.yr)
        """

        self.ICarrays = False
        self.Np = 0
        self.Nsnap = 0
        self.timestep = 0
        self.running = False
        # positions and velocities have dimension (Nstep, Npart, 3)
        self.positions = None
        self.velocities = None
        # masses are (Npart)
        self.masses = None
        self.times = None
        self.extra_force_functions = []
        self.extra_timedependent_force_functions = []
        self.extra_velocitydependent_force_functions = []
        # Things can come in in various units, but use these internally
        self.lenunit = u.kpc
        self.velunit = u.km/u.s
        self.massunit = u.Msun
        self.timeunit = u.Myr
        self.accelunit = self.velunit / self.timeunit
        # Parameters. For historic reasons, these go in params dict.
        # Use the set_ methods which sanity check the inputs
        self.params = {}
        self.set_dt(dt)
        self.set_eps(eps)
        self.set_algorithm(algorithm)
        # Useful information to have stored while in the middle of generating a movie
        # so we don't need to keep figuring out what we're plotting
        self._plot_parms = None

      
    def set_dt(self, dt):
        """Sets the simulation time step.
        
        Parameters
        ----------
        dt : Quantity with dimensions of time
            Time step length
            
        Raises
        ------
        ValueError
            If dt does not have dimensions of time.
            
        See also
        --------
        :meth:`~gravhopper.Simulation.get_dt` : Returns simulation time step.
        """

        try:
            # Make sure that it has dimensions of time
            _ = dt.to(u.Myr)
        except u.UnitConversionError:
            raise ValueError("dt must have dimensions of time.")
            
        self.params['dt'] = dt
        return

        
    def get_dt(self):
        """Returns simulation time step.
        
        Returns
        -------
        dt : Quantity
            Time step length

        See also
        --------
        :meth:`~gravhopper.Simulation.set_dt` : Sets simulation time step.
        """
        return self.params['dt']
        
        
        
    def set_eps(self, eps):
        """Sets the simulation gravitational softening length.
        
        Parameters
        ----------
        eps : Quantity with dimensions of length
            Gravitational softening length
            
        Raises
        ------
        ValueError
            If eps does not have dimensions of length.

        See also
        --------
        :meth:`~gravhopper.Simulation.get_eps` : Returns gravitational softening length.
        """

        try:
            # Make sure that it has dimensions of length
            _ = eps.to(u.kpc)
        except u.UnitConversionError:
            raise ValueError("eps must have dimensions of length.")
            
        self.params['eps'] = eps
        return

        
    def get_eps(self):
        """Returns simulation gravitational softening length.
        
        Returns
        -------
        eps : Quantity
            Gravitational softening length

        See also
        --------
        :meth:`~gravhopper.Simulation.set_eps` : Sets gravitational softening length.
        """
        return self.params['eps']
        
        
    def set_algorithm(self, algorithm):
        """Sets the simulation gravitational force algorithm.
        
        Parameters
        ----------
        algorithm : str ('tree' or 'direct')
            Direct summation or Barnes-Hut tree algorithm
            
        Raises
        ------
        ValueError
            If algorithm is not 'tree' or 'direct'.

        See also
        --------
        :meth:`~gravhopper.Simulation.get_algorithm` : Returns current gravitational force algorithm.
        """
        
        if algorithm in ('tree', 'direct'):
            self.params['algorithm'] = algorithm
        else:
            raise ValueError("algorithm must be 'tree' or 'direct'.")
        return

        
    def get_algorithm(self):
        """Returns current simulation gravitational algorithm.
        
        Returns
        -------
        algorithm : str
            'direct' for direct summation, 'tree' for Barnes-Hut tree

        See also
        --------
        :meth:`~gravhopper.Simulation.set_algorithm` : Sets the gravitational force algorithm.
        """
        return self.params['algorithm']
        
        
    def run(self, N=1):
        """Run N timesteps of the simulation. Will either initialize a simulation that has
        not yet been run, or continue from the last snapshot if it has.
        
        Parameters
        ----------
        N : int
            Number of time steps to perform
        """
        
        # Initialize if first timestep, expand arrays for output if not
        if self.running==False:
            self.init_run(N)
        else:
            self.Nsnap += N
            # Expand arrays
            extra_vec = np.zeros((N, self.Np, 3))
            extra_scalar = np.zeros((N, self.Np))
            extra_single = np.zeros((N))
            self.positions = np.concatenate((self.positions,extra_vec*self.lenunit), axis=0)
            self.velocities = np.concatenate((self.velocities,extra_vec*self.velunit), axis=0)
            self.times = np.concatenate((self.times,extra_single*self.timeunit), axis=0)
            
        # Perform time steps
        for i in range(self.timestep, self.timestep+N):
            self.timestep += 1
            self.perform_timestep()
            self.times[self.timestep] = self.times[self.timestep-1] + self.params['dt']


            
    def init_run(self, Nsnap=None):
        """Initialize an N-body run."""
        
        if self.ICarrays==False:
            raise UninitializedSimulationException
            
        self.Nsnap = Nsnap+1
        self.Np = len(self.ICarrays['pos'])
        # Create the pos, vel, and mass arrays and put ICs in index 0
        self.positions = np.zeros((self.Nsnap, self.Np, 3)) * self.lenunit
        self.velocities = np.zeros((self.Nsnap, self.Np, 3)) * self.velunit
        self.masses = np.zeros((self.Np)) * self.massunit
        self.times = np.zeros((self.Nsnap)) * self.timeunit
        
        self.positions[0,:,:] = self.ICarrays['pos']
        self.velocities[0,:,:] = self.ICarrays['vel']
        self.masses[:] = self.ICarrays['mass']
        
        self.running = True
        
        
        
    def reset(self):
        """Reset a simulation to the state before init_run() was called. Initial conditions
        and external forces are preserved."""

        self.running = False
        self.positions = None
        self.velocities = None
        self.masses = None
        self.times = None
        self.Nsnap = 0
        self.timestep = 0
        
        
        

    def snap(self, step):
        """Return the given snapshot.
        
        Parameters
        ----------
        step : int
            Time step to return
            
        Returns
        -------
        snap : dict
            snap['pos'] is an (Np,3) array of positions
            snap['vel'] is an (Np,3) array of velocities
            snap['mass'] is a length-Np array of masses
        """
        return {'pos':self.positions[step, :, :], 'vel':self.velocities[step, :, :], \
            'mass':self.masses[:]}

                
    def current_snap(self):
        """Return the current snapshot.
        
        Returns
        -------
        snap : dict
            * **snap['pos']** is an (Np,3) array of positions
            * **snap['vel']** is an (Np,3) array of velocities
            * **snap['mass']** is a length-Np array of masses
        """        
        return self.snap(self.timestep)
            
    def prev_snap(self):
        """Return the snapshot before the current snapshot.
        
        Returns
        -------
        snap : dict
            * **snap['pos']** is an (Np,3) array of positions
            * **snap['vel']** is an (Np,3) array of velocities
            * **snap['mass']** is a length-Np array of masses
        """        
        return self.snap(self.timestep-1)


    def perform_timestep(self):
        """Advance the N-body simulation by one snapshot using a DKD leapfrog integrator."""
        # drift-kick-drift leapfrog:
        # half-step drift
        self.current_snap()['pos'][:] = self.prev_snap()['pos'] + 0.5 * self.prev_snap()['vel'] * self.params['dt']
        # full-step kick
        # For time-dependent forces, acceleration is calculated on the half step
        kick_time = self.times[self.timestep-1] + 0.5*self.params['dt']
        accelerations = self.calculate_acceleration(time=kick_time)
        self.current_snap()['vel'][:] = self.prev_snap()['vel'] + accelerations * self.params['dt']
        # half-step drift
        self.current_snap()['pos'][:] += 0.5 * self.current_snap()['vel'] * self.params['dt']
        
        
    def calculate_acceleration(self, time=None):
        """Calculate acceleration for particle positions at current_snap() due to gravitational N-body force, plus any external
        forces that have been added.
         
        Parameters
        ----------
        time : Quantity
            Time at which to calculate any time-dependent external forces. Default: None
            
        Returns
        -------
        acceleration : array
            An (Np,3) numpy array of the acceleration vector calculated for each particle
         
        Raises
        ------
        UnknownAlgorithmException
            If the algorithm parameter is not 'tree' or 'direct'
        """
        
        # gravity
        # If there is only one particle, there is no force, and the C code will give
        # a divide by zero... so just force a zero.
        if self.Np > 1:
            if self.params['algorithm']=='direct':
                nbody_gravity = jbgrav.direct_summation(self.current_snap(), self.params['eps'])
            elif self.params['algorithm']=='tree':
                nbody_gravity = jbgrav.tree_force(self.current_snap(), self.params['eps'])
            else:
                raise UnknownAlgorithmException()
        else:
            nbody_gravity = np.zeros((self.Np, 3)) * self.accelunit
            
        # any extra forces that have been added. Note that accelerations are computed
        # before the kick, so we need to use the previous snap's velocity in case there
        # are velocity-dependent forces
        extra_accel = self.calculate_extra_acceleration(self.current_snap()['pos'], nbody_gravity, \
            time=time, vel=self.prev_snap()['vel'])
        totaccel = nbody_gravity + extra_accel
        
        return totaccel
        
        
    def calculate_extra_acceleration(self, pos, template_array, time=None, vel=None):
        """Calculates the acceleration just due to external added forces."""
        extaccel = np.zeros_like(template_array)
        for fn, args in self.extra_force_functions:
            extaccel += fn(pos, args)
        # and the time-dependent extra forces
        for fn, args in self.extra_timedependent_force_functions:
            extaccel += fn(pos, time, args)
        # and the velocity-dependent extra forces
        for fn, args in self.extra_velocitydependent_force_functions:
            extaccel += fn(pos, vel, args)
        return extaccel


    def gala_potential_wrapper(self, galapot, pos, time=None):
        """Wrapper to calculate acceleration from a gala Potential object."""
        if time is None:
            accel = galapot.acceleration(pos.T).T
        else:
            accel = galapot.acceleration(pos.T, t=time).T
            
        return accel
        
        
    def galpy_potential_wrapper(self, galpypot, pos, time=None):
        """Wrapper to calculate acceleration from a galpy potential object."""

        # Galpy works exclusively in cylindrical coordinate frame, so we need to
        # convert there and back again, a vector's tale.
        R, phi, z = galpy.util.coords.rect_to_cyl(pos[:,0], pos[:,1], pos[:,2])
        
        # Some galpy potentials work on arrays of coordinates, but not all. For
        # potentials where it doesn't, loop through each position (note: much slower,
        # so only do those where I know it doesn't work).
        single_potentials = [galpy.potential.RazorThinExponentialDiskPotential]
        if any([isinstance(galpypot, s) for s in single_potentials]):
            # Do them one by one
            if(len(pos.shape)) > 1:
                Npos = pos.shape[0]
            else:
                Npos = 1
            
            accel_R = np.zeros((Npos)) * self.accelunit
            accel_phi = np.zeros((Npos)) * self.accelunit
            accel_z = np.zeros((Npos)) * self.accelunit
        
            for parti in range(Npos):
                if time is None:
                    accel_R[parti] = galpy.potential.evaluateRforces(galpypot, R[parti], z[parti], phi=phi[parti])
                    # You might think that a function called evaluatephiforces would return
                    # the forces in the phi direction. You would be wrong.
                    # It actually returns dPhi/dphi, which is R times the actual phi force.
                    # So we need to divide by R to get a physical force that we can transform
                    # as a vector.
                    accel_phi[parti] = galpy.potential.evaluatephiforces(galpypot, R[parti], z[parti], phi=phi[parti]) / R[parti]
                    accel_z[parti] = galpy.potential.evaluatezforces(galpypot, R[parti], z[parti], phi=phi[parti])
                else:
                    accel_R[parti] = galpy.potential.evaluateRforces(galpypot, R[parti], z[parti], phi=phi[parti], t=time)
                    # See above.
                    accel_phi[parti] = galpy.potential.evaluatephiforces(galpypot, R[parti], z[parti], phi=phi[parti], t=time) / R[parti]
                    accel_z[parti] = galpy.potential.evaluatezforces(galpypot, R[parti], z[parti], phi=phi[parti], t=time)

            ax, ay, az = galpy.util.coords.cyl_to_rect_vec(accel_R, accel_phi, accel_z, phi=phi)
            
        else:
            # Do it vectorized
            if time is None:
                accel_R = galpy.potential.evaluateRforces(galpypot, R, z, phi=phi)
                # You might think that a function called evaluatephiforces would return
                # the forces in the phi direction. You would be wrong.
                # It actually returns dPhi/dphi, which is R times the actual phi force.
                # So we need to divide by R to get a physical force that we can transform
                # as a vector.
                accel_phi = galpy.potential.evaluatephiforces(galpypot, R, z, phi=phi) / R
                accel_z = galpy.potential.evaluatezforces(galpypot, R, z, phi=phi)
                ax, ay, az = galpy.util.coords.cyl_to_rect_vec(accel_R, accel_phi, accel_z, phi=phi)
            else:
                accel_R = galpy.potential.evaluateRforces(galpypot, R, z, phi=phi, t=time)
                # See above.
                accel_phi = galpy.potential.evaluatephiforces(galpypot, R, z, phi=phi, t=time) / R
                accel_z = galpy.potential.evaluatezforces(galpypot, R, z, phi=phi, t=time)
                ax, ay, az = galpy.util.coords.cyl_to_rect_vec(accel_R, accel_phi, accel_z, phi=phi)
            
        accel = np.vstack((ax,ay,az)).T
        
        return accel
        

    def galpy_dissipativeforce_wrapper(self, galpypot, pos, vel=None):
        """Wrapper to calculate acceleration from a galpy DissipativeForce object."""

        # Galpy works exclusively in cylindrical coordinate frame, so we need to
        # convert there and back again, a vector's tale.
        R, phi, z = galpy.util.coords.rect_to_cyl(pos[:,0], pos[:,1], pos[:,2])
        vR, vphi, vz = galpy.util.coords.rect_to_cyl_vec(vel[:,0], vel[:,1], vel[:,2], pos[:,0], pos[:,1], pos[:,2])
        
        # Some galpy potentials work on arrays of coordinates, but not all. For
        # potentials where it doesn't, loop through each position (note: much slower,
        # so only do those where I know it doesn't work).
        single_potentials = [galpy.potential.ChandrasekharDynamicalFrictionForce]
        if any([isinstance(galpypot, s) for s in single_potentials]):
            # Do them one by one
            if(len(pos.shape)) > 1:
                Npos = pos.shape[0]
            else:
                Npos = 1
            
            accel_R = np.zeros((Npos)) * self.accelunit
            accel_phi = np.zeros((Npos)) * self.accelunit
            accel_z = np.zeros((Npos)) * self.accelunit
        
            for parti in range(Npos):
                veli = [vR[parti], vphi[parti], vz[parti]]
                accel_R[parti] = galpy.potential.evaluateRforces(galpypot, R[parti], z[parti], phi=phi[parti], v=veli)
                # You might think that a function called evaluatephiforces would return
                # the forces in the phi direction. You would be wrong.
                # It actually returns dPhi/dphi, which is R times the actual phi force.
                # So we need to divide by R to get a physical force that we can transform
                # as a vector.
                accel_phi[parti] = galpy.potential.evaluatephiforces(galpypot, R[parti], z[parti], phi=phi[parti], v=veli) / R[parti]
                accel_z[parti] = galpy.potential.evaluatezforces(galpypot, R[parti], z[parti], phi=phi[parti], v=veli)

            ax, ay, az = galpy.util.coords.cyl_to_rect_vec(accel_R, accel_phi, accel_z, phi=phi)
            
        else:
            # Do it vectorized
            vel_cyl = np.vstack((vR, vphi, vz))
            
            accel_R = galpy.potential.evaluateRforces(galpypot, R, z, phi=phi, v=vel_cyl)
            # You might think that a function called evaluatephiforces would return
            # the forces in the phi direction. You would be wrong.
            # It actually returns dPhi/dphi, which is R times the actual phi force.
            # So we need to divide by R to get a physical force that we can transform
            # as a vector.
            accel_phi = galpy.potential.evaluatephiforces(galpypot, R, z, phi=phi, v=vel_cyl) / R
            accel_z = galpy.potential.evaluatezforces(galpypot, R, z, phi=phi, v=vel_cyl)
            ax, ay, az = galpy.util.coords.cyl_to_rect_vec(accel_R, accel_phi, accel_z, phi=phi)
            
        accel = np.vstack((ax,ay,az)).T
        
        return accel



    def add_external_force(self, fn, args=None):
        """Add an external position-dependent force to the simulation.
 
        Forces can be in the form of:
        
        1. A function that takes two arguments: an (Np,3) array of positions
           (must be Quantities) that contains the positions where the accelerations
           are to be calculated, and an additional argument that is a dictionary
           containing any extra parameters you need.

        2. A galpy potential object.

        3. A gala Potential object.

        You may add as many external forces as you want - they will be summed
        together along with the N-body force.

        Parameters
        ----------
        fn : function or galpy potential.Potential object or gala potential.PotentialBase object
            External force function to add
        args : dict
            Any extra parameters that should be passed to the function when it is called
            
        Example
        -------
        This function calculates an external force from a single point source::

            def my_point_source_force(pos, args):
                 # acceleration is G M / r^2
                 GM = const.G * args['mass']
                 d_pos = pos - args['pos']
                 r2 = (d_pos**2).sum(axis=1)
                 rhat = d_pos / np.sqrt(r2)
                 return rhat * GM / r2

        To add the force from a particle at 10kpc on the x-axis with
        a mass of 1e8 Msun::
        
            mysimulation = Simulation()
            mysimulation.add_external_force(my_point_source_force, {'mass':1e8*u.Msun,
              'pos':np.array([10,0,0])*u.kpc})
          
        
        """    
              
        if isinstance(fn, list):
            # Probably a galpy combined potential. Add each one individually.
            for item in fn:
                self.add_external_force(item, args)
            return
        
        if USE_GALA:
            # Check if it's a gala potential
            if isinstance(fn, gala.potential.PotentialBase):
                gala_fn = lambda x, a: self.gala_potential_wrapper(fn, x)
                self.extra_force_functions.append( (gala_fn, args) )
                return
                
        if USE_GALPY:
            # Check if it's a galpy potential
            if isinstance(fn, galpy.potential.Potential):
                galpy_fn = lambda x, a: self.galpy_potential_wrapper(fn, x)
                self.extra_force_functions.append( (galpy_fn, args) )
                return

        # If it hasn't returned before now, it's presumably just a function
        self.extra_force_functions.append( (fn,args) )



    def add_external_timedependent_force(self, fn, args=None):
        """Add an external time-dependent force to the simulation.
         
        Forces can be in the form of:
        
        1. A function that takes three arguments: an (Np,3) array of positions (must be
           Quantities) that contains the positions where the accelerations are to be 
           calculated, a scalar time, and an additional argument that is a dictionary 
           containing any extra parameters you need.
                
        2. A galpy potential object.
           
        3. A gala Potential object.

        You may add as many external forces as you want - they will be summed
        together along with the N-body force.
        
              
        Parameters
        ----------
        fn : function or galpy potential.Potential object or gala potential.PotentialBase object
            External force function to add
        args : dict
            Any extra parameters that should be passed to the function when it is called              


        Example
        -------
        
        This function calculates an external force
        from a single point source whose mass oscilllates in time::
        
            def my_oscillating_point_source_force(pos, time, args):
                 # args has 3 parameters:
                 #    args['pos']: position of mass
                 #    args['massamplitude']: amplitude of oscillating mass (should have mass units)
                 #    args['period']: period of oscillation of mass (should have time units)
                 # acceleration is G M / r^2
                 GM = const.G * args['massamplitude'] * (np.cos(2. * np.pi * time / args['period']) + 1.)
                 d_pos = pos - args['pos']
                 r2 = (d_pos**2).sum(axis=1)
                 rhat = d_pos / np.sqrt(r2)
                 return rhat * GM / r2
                
        To add the force from a particle at 10kpc on the x-axis with
        a mass of 10\ :sup:`8` M\ :sub:`sun` that oscillates every 100 Myr::
        
            mysimulation = Simulation()
            mysimulation.add_external_timedependent_force(my_point_source_force, {'massamplitude':1e8*u.Msun,
              'period':100*u.Myr, 'pos':np.array([10,0,0])*u.kpc})
        
        """

        if isinstance(fn, list):
            # Probably a galpy combined potential. Add each one individually.
            for item in fn:
                self.add_external_timedependent_force(item, args)
            return
        
        if USE_GALA:
            # Check if it's a gala potential
            if isinstance(fn, gala.potential.PotentialBase):
                gala_fn = lambda x, t, a: self.gala_potential_wrapper(fn, x, time=t)
                self.extra_timedependent_force_functions.append( (gala_fn, args) )
                return
                
        if USE_GALPY:
            # Check if it's a galpy potential
            if isinstance(fn, galpy.potential.Potential):
                galpy_fn = lambda x, t, a: self.galpy_potential_wrapper(fn, x, time=t)
                self.extra_timedependent_force_functions.append( (galpy_fn, args) )
                return

        # If it hasn't returned before now, it's presumably just a function
        self.extra_timedependent_force_functions.append( (fn,args) )


    def add_external_velocitydependent_force(self, fn, args=None):
        """Add an external velocity-dependent force to the simulation.
         
        Forces can be in the form of:
        
        1. A function that takes three arguments: an (Np,3) array of positions (must
           be Quantities) that contains the positions where the accelerations
           are to be calculated, an (Np,3) array of velocities (must be Quantities),
           and an additional argument that is a dictionary containing any extra
           parameters you need.
        2. A galpy potential object (must derive from galpy.potential.DissipativeForce).

        You may add as many external forces as you want - they will be summed
        together along with the N-body force.
        
        Parameters
        ----------
        fn : function or galpy.potential.DissipativeForcePotential object
            External force function to add
        args : dict
            Any extra parameters that should be passed to the function when it is called

        Example
        -------
        This function adds on an external force that goes in the opposite directly of the
        current velocity of every particle with magnitude abs(velocity) / timescale given in args::
         
            def my_friction_force(pos, vel, args):
               # args has 1 parameter:
               #    args['t0']:  timescale (should have time units)
               velmag = np.sqrt(np.sum(vel**2, axis=1))
               forcemag = velmag / args['t0']
               forcearray = -vel/velmag[:,np.newaxis] * forcemag[:,np.newaxis]
               return forcearray
           
        Then to add a force that slows all particles down on a 100 Myr timescale::
         
            mysimulation = Simulation()
            mysimulation.add_external_velocitydependent_force(my_friction_force, {'t0':100*u.Myr})                

        """

        if isinstance(fn, list):
            # Probably a galpy combined potential. Add each one individually.
            for item in fn:
                self.add_external_velocitydependent_force(item, args)
            return
        
        if USE_GALPY:
            # Check if it's a galpy velocity-dependent potential
            if isinstance(fn, galpy.potential.DissipativeForce.DissipativeForce):
                galpy_fn = lambda x, v, a: self.galpy_dissipativeforce_wrapper(fn, x, vel=v)
                self.extra_velocitydependent_force_functions.append( (galpy_fn, args) )
                return

        # If it hasn't returned before now, it's presumably just a function
        self.extra_velocitydependent_force_functions.append( (fn,args) )


    def nrows(self, array):
        """Return the number of rows in an array or scalar."""
        return array.shape[0] if array.ndim>1 else 1


    def add_IC(self, newIC):
        """Adds particles to the initial conditions.
        
        Parameters
        ----------
        newIC : dict
           Properties of new particles to add. Must have the following key-value pairs:
           
           * **pos:** an array of positions                
           * **vel:** an array of velocities
           * **mass:** an array of masses
                
           All should be astropy Quantities, with shape (Np,3) or just (3) if a single particle.

        Example
        -------
        Create a simulation object whose initial conditions consist of a
        particle of mass 10\ :sup:`8` M\ :sub:`sun` at a position 10 kpc from the origin on
        the x-axis, and a velocity of 200 km/s in the positive y-direction::
         
            sim = Simulation()
            sim.add_IC( {'pos':np.array([10,0,0])*u.kpc, 'vel':np.array([0,200,0])*u.km/u.s, 
                'mass':np.array([1e8])*u.Msun} )
            
        See Also
        --------
        IC : Class containing static functions that generate initial conditions that can be added using add_IC().
        """

        #sanity check that all pieces are there and have the same number of particles
        if 'pos' not in newIC:
            raise ICException("Missing 'pos' key in initial conditions function.")
        if 'vel' not in newIC:
            raise ICException("Missing 'vel' key in initial conditions function.")
        if 'mass' not in newIC:
            raise ICException("Missing 'mass' key in initial conditions function.")
        npos = self.nrows(newIC['pos'])
        nvel = self.nrows(newIC['vel'])
        nmass = len(newIC['mass'])

        if (npos != nvel) | (npos != nmass):
            raise ICException('Inconsistent number of particles in initial conditions function.')
        if self.ICarrays==False:
            # need to initialize arrays
            self.ICarrays = {}
            self.ICarrays['pos'] = newIC['pos']
            self.ICarrays['vel'] = newIC['vel']
            self.ICarrays['mass'] = newIC['mass']
        else:
            self.ICarrays['pos'] = np.vstack( (self.ICarrays['pos'], newIC['pos']) )
            self.ICarrays['vel'] = np.vstack( (self.ICarrays['vel'], newIC['vel']) )
            self.ICarrays['mass'] = np.hstack( (self.ICarrays['mass'], newIC['mass']) )


    def pyn_snap(self, timestep=None):
        """Return snapshot given by the timestep as a pynbody SimSnap.
        
        Parameters
        ----------
        timestep : int
            Snapshot number to return. Returns final snapshot if timestep is None (default: None)
        
        Returns
        -------
        sim : pynbody SimSnap
            Snapshot
            
        Notes
        -----
        Returned snapshot will have length units of Unit("kpc"), velocity units of Unit("km s**-1"),
        and mass units of Unit("Msol").
        
        Raises
        ------
        ExternalPackageException
            If pynbody could not be imported.
        
        """
        
        if USE_PYNBODY:
            if timestep is None:
                timestep = self.timestep
        
            # I don't have a good converter between astropy units and pynbody units. Pynbody
            # to astropy works okay via string, but not vice versa. Position should be fine,
            # since it will very likely be a base unit, but velocity will certainly be composite,
            # and mass might be too if it was calculated, so manually force them to km/s and Msun.
            pyn_pos_unit = pyn.units.Unit("kpc")
            ap_pos_unit = u.kpc
            pyn_vel_unit = pyn.units.Unit("km s**-1")
            ap_vel_unit = u.km / u.s
            pyn_mass_unit = pyn.units.Unit("Msol")
            ap_mass_unit = u.Msun
                        
            sim = pyn.new(self.Np)
            sim['pos'] = pyn.array.SimArray(self.snap(timestep)['pos'].to(ap_pos_unit).value,\
                pyn_pos_unit)
            sim['vel'] = pyn.array.SimArray(self.snap(timestep)['vel'].to(ap_vel_unit).value, \
                pyn_vel_unit)
            sim['mass'] = pyn.array.SimArray(self.snap(timestep)['mass'].to(ap_mass_unit).value, \
                pyn_mass_unit)
            sim['eps'] = pyn.array.SimArray(self.params['eps'].to(ap_pos_unit).value, pyn_pos_unit)
            
            return sim
        else:
            raise ExternalPackageException("Could not import pynbody to use pyn_snap().")
            
            
    def plot_particles(self, parm='pos', coords='xy', snap='final', xlim=None, ylim=None, \
        s=0.2, unit=None, ax=None, timeformat='{0:.1f}', nolabels=False, particle_range=None, **kwargs):
        """
        Scatter plot of particle positions or velocities for a snapshot.
    
        Parameters
        ----------
        parms : str
            'pos' or 'vel' to plot particle positions or velocities (default: 'pos')
        coords : str
            'xy', 'xz', or 'yz' to plot the different Cartesian projections (default 'xy')
        snap : int or str
            Which snapshot to plot. Either a snapshot number, 'final' for the last snapshot,
            or 'IC' for the initial conditions (default: 'final')
        xlim : array-like, optional
            x limits of plot
        ylim : array-like, optional
            y limits of plot
        s : float
            Scatter plot point size (default: 0.2)
        unit: astropy Unit, optional
            Plot the quantities in the given unit system. The default is whatever units
            the positions or velocities array is in, which is kpc or km/s by default but
            can be changed.
        ax : matplotlib Axis, optional
            Axis on which to plot. If missing or None, uses plt.subplot(111, aspect=1.0) to
            create a new Axis object.
        timeformat : str or False
            Format string for snapshot time in title, or False for no title (default: '{0:.1f}')
        nolabels : bool, optional
            If True, do not label x and y axes (default: False)
        particle_range: array-like, optional
            Only plot particles with indices in the slice particle_range[0]:particle_range[1]
        **kwargs : dict
            All additional keyword arguments are passed onto plt.scatter()
            
        Returns
        -------
        scatter : matplotlib PathCollection
            The scatter plot
        """

        # Create axis if necessary.
        if ax is None:
            ax = plt.subplot(111, aspect=1.0)
            
        # Get the things that will be plotted
        if parm=='pos':
            data = self.positions
        elif parm=='vel':
            data = self.velocities
        else:
            raise ValueError("parm must be 'pos' or 'vel'")
            
        # Figure out coordinate indices
        if len(coords) != 2:
            raise ValueError("coords must have length 2")
        if (coords[0] not in 'xyz') or (coords[1] not in 'xyz'):
            raise ValueError("coords characters must be x, y, or z")
        xindex = ord(coords[0]) - ord('x')
        yindex = ord(coords[1]) - ord('x')
        
        # Figure out snapshot number
        if snap=='final':
            snapnum = self.timestep
        elif snap=='IC':
            snapnum = 0
        else:
            # it's the snapshot number, hopefully
            snapnum = snap
            
        # If unit is None, get default unit
        if unit is None:
            unit = data.unit
            
        # Full range of particles if not specified
        if particle_range is None:
            particle_range = (0, self.Np)
        
        # Make the plot!
        output = ax.scatter(data[snapnum, particle_range[0]:particle_range[1] ,xindex].to(unit).value,\
            data[snapnum, particle_range[0]:particle_range[1], yindex].to(unit).value, s=s, **kwargs)
            
        # Store things we'll need for following frames if making a movie
        self._plot_parms = {'data_parm':parm, 'particle_range':particle_range, 'xindex':xindex, \
            'yindex':yindex, 'unit':unit}
            
        # Set ranges, axis labels, and title
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if nolabels == False:
            if parm=='pos':
                xlabel = coords[0]
                ylabel = coords[1]
            elif parm=='vel':
                xlabel = 'v_'+coords[0]
                ylabel = 'v_'+coords[1]
            
            ax.set_xlabel('${0}$ ({1})'.format(xlabel, str(unit)))
            ax.set_ylabel('${0}$ ({1})'.format(ylabel, str(unit)))
        self._plot_particles_settitle(ax, snapnum, timeformat)
                                
        return output
        

    def _plot_particles_settitle(self, ax, snapnum, timeformat=False):
        """Utility routine to set title of axis to the time of snapnum using the given format."""
        if timeformat != False:
            ax.set_title(timeformat.format(self.times[snapnum]))
            
        
    def _plot_particles_setoffsets(self, scatterplot, snapnum):
        """Update a previously-made plot_particles() with a new snapshot number."""
        
        # Get the things that will be plotted
        if self._plot_parms['data_parm']=='pos':
            data = self.positions
        elif self._plot_parms['data_parm']=='vel':
            data = self.velocities
        else:
            raise ValueError("data_parm must be 'pos' or 'vel'")
            
        xdat = data[snapnum, self._plot_parms['particle_range'][0]:self._plot_parms['particle_range'][1], self._plot_parms['xindex']].to(self._plot_parms['unit']).value
        ydat = data[snapnum, self._plot_parms['particle_range'][0]:self._plot_parms['particle_range'][1], self._plot_parms['yindex']].to(self._plot_parms['unit']).value
                            
        scatterplot.set_offsets(np.c_[xdat, ydat])
        
        return
        
        
    
    def movie_particles(self, fname, fps=25, ax=None, skip=None, timeformat='{0:.1f}', *args, **kwargs):
        """Create a movie of the particles using the :meth:`~gravhopper.Simulation.plot_particles` function.
        
        Parameters
        ----------
        fname : str
            Movie output file name
        fps : int
            Frames per second (default: 25)
        ax : matplotlib Axis, optional
            Axis on which to plot. If missing or None, uses plt.subplot(111, aspect=1.0) to
            create a new Axis object and closes the figure after the movie is made.
        skip : int, optional
            Skip every N frames (e.g. skip=5 only has 1/5th of the full number of frames).
        xlim : array-like, optional
            x limits of the movie (default: encompasses the full extent any particle
            reaches)
        ylim : array-like, optional
            y limits of the movie (default: encompasses the full extent any particle
            reaches)
        *args : object
            Passed through to :meth:`~gravhopper.Simulation.plot_particles`
        **kwargs : dict
            Passed through to :meth:`~gravhopper.Simulation.plot_particles`
        """
        
        # Create axis if necessary.
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect=1.0)
            close_plot = True
        else:
            fig = ax.get_figure()
            close_plot = False
            
        # If skip is None, equivalent to 1
        if skip is None:
            skip = 1
            
        # Initial frame
        particles = self.plot_particles(*args, ax=ax, snap='IC', timeformat=timeformat, **kwargs)
        
        # Default x and y limits are min/max over the whole simulation
        # Get the things that will be plotted
        if self._plot_parms['data_parm']=='pos':
            data = self.positions
        elif self._plot_parms['data_parm']=='vel':
            data = self.velocities
        xdat = data[:, self._plot_parms['particle_range'][0]:self._plot_parms['particle_range'][1], self._plot_parms['xindex']].to(self._plot_parms['unit']).value
        ydat = data[:, self._plot_parms['particle_range'][0]:self._plot_parms['particle_range'][1], self._plot_parms['yindex']].to(self._plot_parms['unit']).value
        if 'xlim' not in kwargs:
            kwargs['xlim'] = [np.min(xdat), np.max(xdat)]
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' not in kwargs:
            kwargs['ylim'] = [np.min(ydat), np.max(ydat)]
            ax.set_ylim(kwargs['ylim'])
            
        
        # Function that updates each frame
        def animate(frame):
            framesnap = frame * skip
            self._plot_particles_setoffsets(particles, framesnap)
            self._plot_particles_settitle(ax, framesnap, timeformat)
            return particles
            
        ms_per_frame = 1000 / fps
        
        anim = FuncAnimation(fig, animate, frames=(self.timestep+1) // skip, interval=ms_per_frame)
        anim.save(fname)
        
        if close_plot:
            plt.close(fig)



        
class IC(object):
    """Namespace for holding static methods that create a variety of initial conditions."""
    
    @staticmethod
    def from_galpy_df(df, N=None, totmass=None, center_pos=None, center_vel=None, force_origin=True):
        """Sample a galpy sphericaldf distribution function object and return as an IC.
        
        Parameters
        ----------
        df : galpy df.sphericaldf object
            Distribution function to sample. Assumed to be in the ro=8, vo=220 unit system.
        N : int
            Number of particles
        totmass : astropy Quantity
            Total mass
        center_pos : 3 element array-like Quantity, optional
            Force the center of mass of the IC to be at this position
        center_vel : 3 element array-like Quantity, optional
            Force the mean velocity of the IC to have this velocity
        force_origin : bool
            Force the center of mass to be at the origin and the mean velocity to be zero;
            equivalent to setting center_pos=[0,0,0]*u.kpc and
            center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
            center_vel is set. If force_origin is True and only one of center_pos or
            center_vel is set, the other is set to zero.
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).
            
        Note
        ----
        Up to at least galpy v1.7, galpy df objects don't fully incorporate astropy Quantity inputs and
        outputs. Therefore, it is recommended that you define any relevant potential using
        the default ro=8, vo=220 unit system, turn off physical output for the potential,
        create the df object from the potential, use ``from_galpy_df()`` to create the ICs,
        and then turn physical output back on again afterwards if needed (see Example). 
        
            
        Example
        -------
        Sample an NFW halo with scale radius 20 kpc, scale amplitude 2x10\ :sup:`11` solar masses, and a maximum
        radius of 1 Mpc with 10,000 particles::
        
              from astropy import units as u
              from galpy import potential, df
    
              NFWamp = 2e11 * u.Msun
              NFWrs = 20 * u.kpc
              ro = 8.
              vo = 220.
              rmax = 1 * u.Mpc
              rmax_over_ro = (rmax/(ro*u.kpc)).to(1).value
              Nhalo = 10000
              NFWpot = potential.NFWPotential(amp=NFWamp, a=NFWrs)
              NFWmass = potential.mass(NFWpot, rmax)
              potential.turn_physical_off(NFWpot)
              NFWdf = df.isotropicNFWdf(pot=NFWpot, rmax=rmax_over_ro)

              halo_IC = IC.from_galpy_df(NFWdf, N=Nhalo, totmass=NFWmass)  
      
              potential.turn_physical_on(NFWpot)   # optional if needed later      
            
        
        Raises
        ------
        ExternalPackageException
            If galpy could not be imported.
        """
        
        if USE_GALPY:
            if not isinstance(df, galpy.df.sphericaldf):
                raise ICException("from_galpy_df() only currently works on galpy.df.sphericaldf objects")
            
            R,vR,vT,z,vz,phi = df.sample(n=N, return_orbit=False)
            # galpy.df absolutely refuses to return physical units. So I will assume
            # that it's returned in "length_unit=8kpc velocity_unit=220km/s" units.
            lenunit = 8*u.kpc
            velunit = 220*u.km/u.s
            
            x, y, z = galpy.util.coords.cyl_to_rect(R, phi, z) * lenunit
            vx, vy, vz = galpy.util.coords.cyl_to_rect_vec(vR, vT, vz, phi) * velunit
            
            m = np.ones((N)) * (totmass/N)
            
            # Force COM and/or COV
            positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
                center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
        
            outIC = {'pos':positions, 'vel':velocities, 'mass': m}
            
            return outIC
            
        else:
            raise ExternalPackageException("Could not import galpy to use from_galpy_df()")
            
            
    @staticmethod
    def from_pyn_snap(pynsnap):
        """Turn a pynbody SimSnap into a set of GravHopper initial conditions.
        
        Parameters
        ----------
        pynsnap : SimSnap
            pynbody snapshot to convert
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).

        Note
        ----
        The IC will be converted into the kpc-km/s-Msun unit system.
           
        Raises
        ------
        ExternalPackageException
            If pynbody could not be imported.        
        """
        
        if USE_PYNBODY:
            # Pynbody units can be complicated things with constants embedded in the
            # unit. Convert them to a simple astropy units system of kpc-km/s-Msun.
            pyn_pos_unit = pyn.units.Unit("kpc")
            ap_pos_unit = u.kpc
            pyn_vel_unit = pyn.units.Unit("km s**-1")
            ap_vel_unit = u.km / u.s
            pyn_mass_unit = pyn.units.Unit("Msol")
            ap_mass_unit = u.Msun
                
            # Grab as numpy arrays and switch from pynsnap units to astropy units
            positions = pynsnap['pos'].in_units(pyn_pos_unit).view(type=np.ndarray) * ap_pos_unit
            velocities = pynsnap['vel'].in_units(pyn_vel_unit).view(type=np.ndarray) * ap_vel_unit
            masses = pynsnap['mass'].in_units(pyn_mass_unit).view(type=np.ndarray) * ap_mass_unit
            
            outIC = {'pos':positions, 'vel':velocities, 'mass':masses}
            
            return outIC
        else:
            raise ExternalPackageException("Could not import pynbody to use from_pyn_snap()")
            

    
    @staticmethod
    def TSIS(N=None, maxrad=None, totmass=None, center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Generate the initial conditions for a Truncated Singular Isothermal Sphere
        (e.g. BT eqs 4.103 and 4.104 with a maximum radius imposed). Note that this is
        not a true equilibrium because of the truncation -- it will be in apporoximate
        equilibrium in the inner regions, at least at first, but not in the outer regions.
         
        Parameters
        ----------
        N : int
            Number of particles
        maxrad : astropy Quantity with length dimensions
            Truncation radius
        totmass : astropy Quantity with mass dimensions
            Total mass
        center_pos : 3 element array-like Quantity, optional
            Force the center of mass of the IC to be at this position
        center_vel : 3 element array-like Quantity, optional
            Force the mean velocity of the IC to have this velocity
        force_origin : bool
            Force the center of mass to be at the origin and the mean velocity to be zero;
            equivalent to setting center_pos=[0,0,0]*u.kpc and
            center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
            center_vel is set. If force_origin is True and only one of center_pos or
            center_vel is set, the other is set to zero.
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            Seed to initialize random number generator to enable repeatable ICs.
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).

        Example
        -------
        To create a truncated singular isothermal sphere with a total mass of 10\ :sup:`11` solar masses,
        a truncation radius of 100 kpc, sampled with 10,000 particles::
        
            particles = IC.TSiS(N=10000, maxrad=100*u.kpc, totmass=1e11*u.Msun)
            
        """

        if (N is None) or (maxrad is None) or (totmass is None):
            raise ICException("TSIS requires N, maxrad, and totmass.")
            
        rng = np.random.default_rng(seed)
            
        sigma = np.sqrt(totmass * const.G / (2 * maxrad)).to(u.km/u.s)
        # generate random coordinates and velocities, and compute particle mass
        radius = rng.uniform(0.0, maxrad.value, size=N) * maxrad.unit
        costheta = rng.uniform(-1.0, 1.0, size=N)
        phi = rng.uniform(0.0, 2.0*np.pi, size=N)
        sintheta = np.sqrt(1.0 - costheta**2)
        x = radius * sintheta * np.cos(phi)
        y = radius * sintheta * np.sin(phi)
        z = radius * costheta
        vx = rng.normal(0.0, sigma.value, size=N) * sigma.unit
        vy = rng.normal(0.0, sigma.value, size=N) * sigma.unit
        vz = rng.normal(0.0, sigma.value, size=N) * sigma.unit
        m = np.ones((N)) * (totmass/N)

        # Force COM and/or COV
        positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
            center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
        
        outIC = {'pos':positions, 'vel':velocities, 'mass': m}

        return outIC


    @staticmethod
    def Plummer(N=None, b=None, totmass=None, center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Generate the initial conditions for an isotropic Plummer model (BT eqs. 4.83 with n=5, 4.92, 2.44b).
        
        Parameters
        ----------
        N : int
            Number of particles
        b : astropy Quantity of dimension length
            Scale radius
        totmass : astropy Quantity of dimensions mass
            Total mass
        center_pos : 3 element array-like Quantity, optional
            Force the center of mass of the IC to be at this position
        center_vel : 3 element array-like Quantity, optional
            Force the mean velocity of the IC to have this velocity
        force_origin : bool
            Force the center of mass to be at the origin and the mean velocity to be zero;
            equivalent to setting center_pos=[0,0,0]*u.kpc and
            center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
            center_vel is set. If force_origin is True and only one of center_pos or
            center_vel is set, the other is set to zero.
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            Seed to initialize random number generator to enable repeatable ICs.
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).
           
        Example
        -------
        To create a Plummer sphere with scale radius 1 pc and a total mass of 10\ :sup:`6` M\ :sub:`sun`
        sampled with 10,000 particles::
        
            particles = IC.Plummer(N=10000, b=1*u.pc, totmass=1e6*u.Msun)
            
        """

        if (N is None) or (b is None) or (totmass is None):
            raise ICException("Plummer requires N, b, and totmass.")

        rng = np.random.default_rng(seed)

        # generate random coordinates and velocities. Uses the law of
        # transformation of probabilities.
        rad_xi = rng.uniform(0.0, 1.0, size=N)
        radius = b / np.sqrt(rad_xi**(-2./3) - 1)
        costheta = rng.uniform(-1.0, 1.0, size=N)
        phi = rng.uniform(0.0, 2.0*np.pi, size=N)
        sintheta = np.sqrt(1.0 - costheta**2)
        x = radius * sintheta * np.cos(phi)
        y = radius * sintheta * np.sin(phi)
        z = radius * costheta

        # need to do the velocity component numerically
        # from Aarseth+ 1974, we want to draw q from q^2 (1-q^2)^(7/2)
        # and then assign the magnitude of v to be
        # v = q sqrt(2) (1 + r^2/b^2)^(-1/4)
        qax = np.arange(0, 1.01, 0.01)
        q_prob = qax**2 * (1. - qax**2)**(3.5)
        q_cumprob = np.cumsum(q_prob) # cumulative probability
        q_cumprob /= q_cumprob[-1]    # normalized correctly to end up at 1
        probtransform = interp1d(q_cumprob, qax)   # reverse interpolation
        # now get the uniform random deviate and transform it
        vel_xi = rng.uniform(0.0, 1.0, size=N)
        q = probtransform(vel_xi)
        velocity = q * np.sqrt(2. * const.G * totmass / b).to(u.km/u.s) * (1. + (radius/b)**2)**(-0.25)
        cosveltheta = rng.uniform(-1.0, 1.0, size=N)
        velphi = rng.uniform(0.0, 2.0*np.pi, size=N)
        sinveltheta = np.sqrt(1.0 - cosveltheta**2)
        vx = velocity * sinveltheta * np.cos(velphi)
        vy = velocity * sinveltheta * np.sin(velphi)
        vz = velocity * cosveltheta

        m = np.ones((N)) * (totmass/N)

        # Force COM and/or COV
        positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
            center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
        
        outIC = {'pos':positions, 'vel':velocities, 'mass': m}

        return outIC


    @staticmethod
    def Hernquist(N=None, a=None, totmass=None, cutoff=10., center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Generate the initial conditions for an isotropic Hernquist model (Hernquist 1990).
        
        Parameters
        ----------
        N : int
            Number of particles
        a : astropy Quantity of dimension length
            Scale radius
        totmass : astropy Quantity of dimension mass
            Total mass
        cutoff : float
            Cut off the distribution at cutoff times the scale radius
        center_pos : 3 element array-like Quantity, optional
            Force the center of mass of the IC to be at this position
        center_vel : 3 element array-like Quantity, optional
            Force the mean velocity of the IC to have this velocity
        force_origin : bool
            Force the center of mass to be at the origin and the mean velocity to be zero;
            equivalent to setting center_pos=[0,0,0]*u.kpc and
            center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
            center_vel is set. If force_origin is True and only one of center_pos or
            center_vel is set, the other is set to zero.
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            Seed to initialize random number generator to enable repeatable ICs.
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).
           
        Example
        -------
        To create a Hernquist sphere with a total mass of 10\ :sup:`10` solar masses, a scale radius
        of 1 kpc, sampled with 10,000 particles::
        
            particles = IC.Hernquist(N=10000, a=1*u.kpc, totmass=1e10*u.Msun)
            
        """
         
        rng = np.random.default_rng(seed)
     
        # Based on equation 10 from Hernquist 1990, I get r/a = 1/(xi^(-1/2) - 1)
        # and xi = r^2 / (1 + r)^2     where r means r/a
        # get positions
        # turn cutoff into max xi
        xi_cutoff = cutoff**2 / ((1. + cutoff)**2)
        rad_xi = rng.uniform(0.0, xi_cutoff, size=N)
        r_over_a = 1./(1./np.sqrt(rad_xi) - 1.)
        radius = r_over_a * a
        costheta = rng.uniform(-1.0, 1.0, size=N)
        phi = rng.uniform(0.0, 2.*np.pi, size=N)
        sintheta = np.sqrt(1.0 - costheta**2)
        x = radius * sintheta * np.cos(phi)
        y = radius * sintheta * np.sin(phi)
        z = radius * costheta

        # For velocities, define f(E) from (17) (as re-expressed by Baes & Dejonghe 2002) and
        # sample the cumulative function for interpolation.
        def fE(E):
            return (np.sqrt(E)*(1-2.*E)*(8.*E*E - 8*E - 3.)/((1.-E)**2) + \
                3.*np.arcsin(np.sqrt(E))/((1.-E)**(5./2)))/(8.*np.sqrt(2)*np.pi**3)
        Eax = np.arange(0.0, 1.0, 0.002)
        cumulative_fE = [integrate.quad(fE, 0.0, Etop)[0] for Etop in Eax]
    
        # binding E must be greater than -potential
        potential = -1./(1. + r_over_a)
        # check and make sure the top of the interpolation bound is okay, otherwise add an extra point
        most_bound_potential = np.max(-potential)
        if most_bound_potential > Eax.max():
            np.append(Eax, most_bound_potential)
            cumulative_fE.append(integrate.quad(fE, 0.0, most_bound_potential)[0])
        cumulative_fE = np.array(cumulative_fE)/np.max(cumulative_fE)
        # build interpolation functions
        Einterp = interp1d(cumulative_fE, Eax)
        inverse_Einterp = interp1d(Eax, cumulative_fE)
    
        # instead of going from 0 to 1, go from 0 to max possible for that radius
        # use inverse interpolation function to find maximum possible xi for a given radius
        max_possible_xi = inverse_Einterp(-potential)
        E_xi = rng.uniform(0.0, max_possible_xi, size=N)
        bindingE = Einterp(E_xi)
        # convert to velocity in real units
        energy_units = const.G * totmass / a
        # E = - 0.5 v^2 - Phi(r)
        # so v = sqrt(2 |E+Phi(r)|)
        velocity = np.sqrt(2. * (-(bindingE+potential))*energy_units).to(u.km/u.s)
        # give it a random direction
        cosveltheta = rng.uniform(-1.0, 1.0, size=N)
        velphi = rng.uniform(0.0, 2.0*np.pi, size=N)
        sinveltheta = np.sqrt(1.0 - cosveltheta**2)
        vx = velocity * sinveltheta * np.cos(velphi)
        vy = velocity * sinveltheta * np.sin(velphi)
        vz = velocity * cosveltheta

        m = np.ones((N)) * (totmass/N)
    
        # Force COM and/or COV
        positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
            center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
        
        outIC = {'pos':positions, 'vel':velocities, 'mass': m}

        return outIC


    @staticmethod
    def expdisk(sigma0=None, Rd=None, z0=None, sigmaR_Rd=None, external_rotcurve=None, N=None, \
        center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Generates initial conditions of an exponential disk with a sech^2 vertical distribution that is
        in (very) approximate equilibrium: rho(R,z) = (sigma0 / 2 z0) exp(-R/Rd) sech^2(z/z0)
        
        Parameters
        ----------
        sigma0 : astropy Quantity with dimensions of surface density
            Central surface density
        Rd : astropy Quantity with dimensions of length
            Radial exponential scale length
        z0 : astropy Quantity with dimensions of length
            Vertical scale height
        sigmaR_Rd : astropy Quantity with dimensions of velocity
            Radial velocity dispersion at R=Rd
        external_rotcurve : function or None
            Function that returns the circular velocity of any external potential that contributes
            to the rotation curve aside from the disk itself. The function should accept input
            as an astropy Quantity of dimension length, and should return an astropy Quantity of
            dimension velocity.
        N : int
            Number of particles
        center_pos : 3 element array-like Quantity, optional
            Force the center of mass of the IC to be at this position
        center_vel : 3 element array-like Quantity, optional
            Force the mean velocity of the IC to have this velocity
        force_origin : bool
            Force the center of mass to be at the origin and the mean velocity to be zero;
            equivalent to setting center_pos=[0,0,0]*u.kpc and
            center_vel=[0,0,0]*u.km/u.s. Default is True unless center_pos and
            center_vel is set. If force_origin is True and only one of center_pos or
            center_vel is set, the other is set to zero.
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            Seed to initialize random number generator to enable repeatable ICs.
            
        Returns
        -------
        IC : dict
           Properties of new particles to add, which sample the given distribution function. Contains
           the following key/value pairs:
           
           * **pos:** an array of positions
           * **vel:** an array of velocities
           * **mass:** an array of masses
           
           Each are astropy Quantities, with shape (Np,3).
           
        Example
        -------
        To create an exponential disk that is in a background logarithmic halo potential that
        generates a flat rotation curve of 200 km/s::
        
            particles = IC.expdisk(N=10000, sigma0=200*u.Msun/u.pc**2, Rd=2*u.kpc,
                z0=0.5*u.kpc, sigmaR_Rd=10*u.km/u.s,
                external_rotcurve=lambda x: 200*u.km/u.s)
                
        """
                
        rng = np.random.default_rng(seed)

        Rd_kpc = Rd.to(u.kpc).value
        totmass = (np.pi * Rd**2 * sigma0).to(u.Msun)

        # cylindrical radius transformation to give an exponential
        Rax = np.arange(0.001*Rd_kpc, 10*Rd_kpc, 0.01*Rd_kpc)
        R_cumprob = Rd_kpc**2 - Rd_kpc*np.exp(-Rax/Rd_kpc)*(Rax+Rd_kpc)
        R_cumprob /= R_cumprob[-1]
        probtransform = interp1d(R_cumprob, Rax)   # reverse interpolation
        # now get the uniform random deviate and transform it
        R_xi = rng.uniform(0.0, 1.0, size=N)
        R = probtransform(R_xi) * u.kpc
        # use random azimuth
        phi = rng.uniform(0.0, 2.0*np.pi, size=N)
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        # get z from uniform random deviate
        z_xi = rng.uniform(0, 1.0, size=N)
        z = 2 * z0 * np.arctanh(z_xi)
        z *= (2 * (rng.uniform(0, 1, size=N) < 0.5)) - 1

        # the velocity dispersions go as:
        #  sigma_R = sigmaR_Rd * exp(-R/Rd)
        #  sigma2_phi = sigmaR^2 * kappa^2 / 4 Omega^2
        #  sigma2_z = pi G Sigma(R) z0 / 2
        # and the mean azimuthal velocity is
        #  <vphi> = vc
        

        def om2(rad):
            # Input must be in kpc but with units stripped, because it's passed into np.derivative.
            y_R = rad/(2.*Rd_kpc)
            # Disk contribution
            omega2 = np.pi * const.G * sigma0 / Rd * (special.iv(0,y_R)*special.kv(0,y_R) -
                    special.iv(1,y_R)*special.kv(1,y_R))
 
            # Halo contribution                   
            if external_rotcurve is not None:
                omega_halo = external_rotcurve(rad*u.kpc) / (rad*u.kpc)
                omega2 += omega_halo**2
                
            return omega2

        Omega2 = om2(R.to(u.kpc).value)
        kappa2 = 4.*Omega2 + R * derivative(om2, R.to(u.kpc).value, 1e-3) / u.kpc

        sigma_R = sigmaR_Rd * np.exp(-R/Rd)
        sigma2_phi = sigma_R**2 * 4 * Omega2 / kappa2
        sigma2_z = np.pi * const.G * z0 * sigma0 * 0.5 * np.exp(-R/Rd)
        vphi_mean = (R * np.sqrt(Omega2)).to(u.km/u.s)

        vphi = np.sqrt(sigma2_phi).to(u.km/u.s) * rng.normal(size=N) + vphi_mean
        vR = sigma_R.to(u.km/u.s) * rng.normal(size=N)
        vx = -vphi * np.sin(phi) + vR * np.cos(phi)
        vy = vphi * np.cos(phi) + vR * np.sin(phi)
        vz = np.sqrt(sigma2_z).to(u.km/u.s) * rng.normal(size=N)

        m = np.ones((N)) * (totmass/N)
        
        # Force COM and/or COV
        positions, velocities = force_centers(np.vstack((x,y,z)).T, np.vstack((vx,vy,vz)).T, \
            center_pos=center_pos, center_vel=center_vel, force_origin=force_origin)
        
        outIC = {'pos':positions, 'vel':velocities, 'mass': m}
        return outIC




        
def force_centers(positions, velocities, center_pos=None, center_vel=None, force_origin=True):
    """Move positions and velocities to have the desired center of mass position and mean velocity.
    
    Parameters
    ----------
    positions : array of Quantities of dimension length
        (Np,3) array of particle positions
    velocities : array of Quantities of dimension velocity
        (Np,3) array of particle velocities
    center_pos : 3 element array-like Quantity, optional
        Force the center of mass of the IC to be at this position
    center_vel : 3 element array-like Quantity, optional
        Force the mean velocity of the IC to have this velocity
    force_origin : bool
        Force the center of mass to be at the origin and the mean velocity to be zero;
        equivalent to setting center_pos=np.array([0,0,0])*u.kpc and
        center_vel=np.array([0,0,0])*u.km/u.s. Default is True unless center_pos and
        center_vel is set. If force_origin is True and only one of center_pos or
        center_vel is set, the other is set to zero.
        
    Returns
    -------
    newpositions : array of Quantities of dimension length
        New shifted positions
    newvelocities : array of Quantities of dimension velocity
        New shifted velocities
    """
    
    newpos = positions
    newvel = velocities
    
    if force_origin:
        if center_pos is None:
            center_pos = np.array([0,0,0])*u.kpc
        if center_vel is None:
            center_vel = np.array([0,0,0])*u.km/u.s
    if center_pos is not None:
        sampled_com = np.mean(positions, axis=0)
        dpos = center_pos - sampled_com
        newpos += dpos
    if center_vel is not None:
        sampled_cov = np.mean(velocities, axis=0)
        dvel = center_vel - sampled_cov
        newvel += dvel 

    return (newpos, newvel)


