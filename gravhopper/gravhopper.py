#!/usr/bin/py

# GravHopper

# This is a very simple gravitational N-body simulator.
# Can take advantage of galpy or gala potentials, and pynbody
# snapshot outputs.

# Written by Jeremy Bailin, University of Alabama
# First initial last name at ua dot edu

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
    """Main class for N-body simulation."""
    
    def __init__(self, dt=1*u.Myr, eps=100*u.pc, algorithm='tree'):
        """Initializes class variables, including defaults.

           Parameters that can be overridden:
            dt: Time step (Astropy Quantity. Default: 1 Myr).
            eps: Gravitational softening length (Astropy Quantity: 100 pc).
            algorithm: N-body force calculation algorithm (default: 'tree').
                         Other options: 'direct'

           Example, to create a new simulation with a timestep of 1 year:
           sim = Simulation(dt=1.0*u.yr)"""

        self.ICarrays = False
        self.IC = False
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
        # Things can come in in various units, but use these internally
        self.lenunit = u.kpc
        self.velunit = u.km/u.s
        self.massunit = u.Msun
        self.timeunit = u.Myr
        self.accelunit = self.velunit / self.timeunit
        # Parameters. For historic reasons, put these in a params dict
        self.params = {'dt':dt, 'eps':eps, 'algorithm':algorithm}
      
        
    def run(self, N=1):
        """Run N timesteps of the simulation. Will either initialize a simulation that has
        not yet been run, or add onto the end of an already-run simulation."""
        
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
        """Return the given snapshot as a dict with entries 'pos', 'vel', and 'mass'."""
        return {'pos':self.positions[step, :, :], 'vel':self.velocities[step, :, :], \
            'mass':self.masses[:]}
                
    def current_snap(self):
        """Return the current snapshot as a dict with entries 'pos', 'vel', and 'mass'."""
        return self.snap(self.timestep)
            
    def prev_snap(self):
        """Return the snapshot before the current snapshot as a dict with entries 'pos', 'vel', and 'mass'."""
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
        """Calculate acceleration due to gravitational N-body force, plus any external
         forces that have been added."""
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
            
        # any extra forces that have been added
        extra_accel = self.calculate_extra_acceleration(self.current_snap()['pos'], nbody_gravity, time)
        totaccel = nbody_gravity + extra_accel
        return totaccel
        
    def calculate_extra_acceleration(self, pos, template_array, time=None):
        """Calculates the acceleration just due to external added forces."""
        extaccel = np.zeros_like(template_array)
        for fn, args in self.extra_force_functions:
            extaccel += fn(pos, args)
        # and the time-dependent extra forces
        for fn, args in self.extra_timedependent_force_functions:
            extaccel += fn(pos, time, args)
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
        
        accel = np.vstack((ax, ay, az)).T
        return accel
        


    def add_external_force(self, fn, args):
        """Add a function that will return the force at a given position,
         which will be added as an external force to the simulation on top
         of the N-body force.
         
         Options are:
           1. A function that takes two arguments: an (Np,3) array of positions
                (must be Quantities) that contains the positions where the accelerations
                are to be calculated, and an additional argument that is a dictionary
                containing any extra parameters you need.
                
           2. A galpy potential object.
           
           3. A gala Potential object.

         You may add as many external forces as you want - they will be summed
         together along with the N-body force.
        
         For example, here is a function that would add on an external force
         from a single point source:
        
         def my_point_source_force(pos, args):
             # acceleration is G M / r^2
             GM = const.G * args['mass']
             d_pos = pos - args['pos']
             r2 = (d_pos**2).sum(axis=1)
             rhat = d_pos / np.sqrt(r2)
             return rhat * GM / r2
        
         Then to add the force from a particle at 10kpc on the x-axis with
         a mass of 1e8 Msun, you would do the following:
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



    def add_external_timedependent_force(self, fn, args):
        """Add a function that will return the force at a given position and time,
         which will be added as an external force to the simulation on top
         of the N-body force.
         
         Options are:
           1. A function that takes three arguments: an (Np,3) array of positions
                (must be Quantities) that contains the positions where the accelerations
                are to be calculated, a scalar time, and an additional argument that is a
                dictionary containing any extra parameters you need.
                
           2. A galpy potential object.
           
           3. A gala Potential object.

         You may add as many external forces as you want - they will be summed
         together along with the N-body force.
        
         For example, here is a function that would add on an external force
         from a single point source whose mass oscilllates in time:
        
         def my_oscillating_point_source_force(pos, time, args):
             # args has 3 parameters:
             #    args['pos']: position of mass
             #    args['massamplitude']: amplitude of oscillating mass (should have mass units)
             #    args['period']: period of oscillation of mass (should have time units)
             # acceleration is G M / r^2
             GM = const * args['massamplitude'] * (np.cos(2. * np.pi * time / args['period']) + 1.)
             d_pos = pos - args['pos']
             r2 = (d_pos**2).sum(axis=1)
             rhat = d_pos / np.sqrt(r2)
             return rhat * GM / r2
                
         Then to add the force from a particle at 10kpc on the x-axis with
         a mass of 1e8 Msol that oscillates every 100 Myr, you would do the following:
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
                gala_fn = lambda x, t, a: self.gala_potential.wrapper(fn, x, time=t)
                self.extra_timedependent_force_functions.append( (gala_fn, args) )
                return
                
        if USE_GALPY:
            # Check if it's a galpy potential
            if isinstance(fn, galpy.potential.Potential):
                galpy_fn = lambda x, t, a: self.galpy_potential.wrapper(fn, x, time=t)
                self.extra_timedependent_force_functions.append( (galpy_fn, args) )
                return

        # If it hasn't returned before now, it's presumably just a function
        self.extra_timedependent_force_functions.append( (fn,args) )




    def nrows(self, array):
        """Return the number of rows in an array or scalar."""
        return array.shape[0] if array.ndim>1 else 1


    def add_IC(self, newIC):
        """Add particles to the initial conditions.
         newIC must be a dict with the following key/value pairs:
           'pos': an array of positions
           'vel': an array of velocities
           'mass': an array of masses
         All should be astropy Quantities, with shape (Np,3) or just (3) if a single particle.

         For example, if you have a Simulation object sim, then to add one
         particle of mass 1e8 Msun at a position 10 kpc from the origin on
         the x-axis, and a velocity of 200 km/s in the positive y-direction:
         
         sim.add_IC( {'pos':np.array([10,0,0])*u.kpc, 'vel':np.array([0,200,0])*u.km/u.s, 
            'mass':np.array([1e8])*u.Msun} )

         See the functions in jbnbody.IC for other examples."""

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
        """Return snapshot given by the timestep as a pynbody snapshot. Gives final
        snapshot if timestep is None."""
        
        if USE_PYNBODY:
            if timestep is None:
                timestep = self.timestep
        
            sim = pyn.new(self.Np)
            sim['pos'] = pyn.array.SimArray(self.snap(timestep)['pos'].value, str(self.snap(timestep)['pos'].unit))
            sim['vel'] = pyn.array.SimArray(self.snap(timestep)['vel'].value, str(self.snap(timestep)['vel'].unit))
            sim['mass'] = pyn.array.SimArray(self.snap(timestep)['mass'].value, str(self.snap(timestep)['mass'].unit))
            sim['eps'] = self.params['eps']
            
            return sim
        else:
            raise ExternalPackageException("Could not import pynbody to use pyn_snap().")
            
            
    def plot_particles(self, parm='pos', coords='xy', snap='final', xlim=None, ylim=None, \
        s=0.2, unit=None, ax=None, timeformat='{0:.1f}', nolabels=False, **kwargs):
        """
        Scatter plot of particle positions or velocities for a snapshot.
    
        Parameters:
            parm:       'pos' or 'vel' to plot particles positions or velocities. Default: 'pos'
            coords:     'xy', 'xz', or 'yz' to plot the various projections. Default: 'xy'
            snap:       Which snapshot to plot. Either a snapshot number, 'final' for the last
                            snapshot, or 'IC' for initial conditions. Default: 'final'
            xlim:       xrange of plot, or None for matplotlib choice. Default: None
            ylim:       yrange of plot, or None for matplotlib choice. Default: None
            s:          Scatter plot size. Default: 0.2
            unit:       Unit for x and y axes. Default: Unit of self.positions or self.velocities.
            ax:         matplotlib Axis to plot on. If None, uses plt.subplot(111, aspect=1.0) to create one.
                            Default: None.
            timeformat: Format string for snapshot time in title, or False for no title. Useful for designating the
                            number of decimals that will make sense, (e.g. '{0:.1f}' for one
                            decimal place). Default: '{0:.1f}'
            nolabels:   Do not label axes. Default: False
            
        Any additional keyword arguments are passed onto plt.scatter()
                        
        Returns the scatter plot.
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
            
        # Make the plot!
        output = ax.scatter(data[snapnum,:,xindex].to(unit).value, data[snapnum,:,yindex].to(unit).value, \
            s=s, **kwargs)
            
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
        if timeformat != False:
            ax.set_title(timeformat.format(self.times[snapnum]))
            
        return output


    def movie_particles(self, fname, fps=25, ax=None, *args, **kwargs):
        """Create a movie of the particles. Uses the plot_particles() function.
        
        Parameters:
            fname:      Movie output file name. Required.
            fps:        Frames per second. Default: 25.
            
        All other parameters are passed through to plot_particles().
        """
        
        # Create axis if necessary.
        if ax is None:
            ax = plt.subplot(111, aspect=1.0)
        fig = ax.get_figure()
            
        # Initial frame
        particles = self.plot_particles(*args, ax=ax, snap='IC', **kwargs)
        
        # Function that updates each frame
        def animate(frame):
            fig.clf()
            # Update particle positions
            particles = self.plot_particles(*args, ax=None, snap=frame, **kwargs)
            return particles
            
        ms_per_frame = 1000 / fps
        
        anim = FuncAnimation(fig, animate, frames=self.timestep+1, interval=ms_per_frame)
        anim.save(fname)



        
class IC(object):
    """Namespace for holding static methods that define various ICs."""
    
    @staticmethod
    def from_galpy_df(df, N=None, totmass=None, center_pos=None, center_vel=None, force_origin=True):
        """Sample a galpy sphericaldf DF object and return as an IC. Arguments:
            N: Number of particles (required)
            totmass: Total mass (astropy Quantity, required)
        Optional:
            center_pos: Force center of mass of simulation to here. Quantity array of size 3.
            center_vel: Force center of mass velocity of simulation to this. Quantity array
                of size 3.
            force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
                and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
                and center_vel is set. If only one of center_mass and center_vel are set,
                and force_origin is True, then the other is set to 0,0,0.
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
    def TSIS(N=None, maxrad=None, totmass=None, center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Returns the positions, velocities, and masses for particles that
         form a truncated singular isothermal sphere (e.g. BT eqs 4.103 and 4.104
         with a maximum radius imposed). Note that this is not a true equilibrium
         because of the truncation -- it will be in apporoximate equilibrium in the 
         inner regions, at least at first, but not in the outer regions.
         
         The parameters are:
           N: number of particles
           maxrad: truncation radius (astropy Quantity)
           totmass: total mass (astropy Quantity)
            center_pos: Force center of mass of simulation to here. Quantity array of size 3. Optional.
            center_vel: Force center of mass velocity of simulation to this. Quantity array
                of size 3. Optional.
            force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
                and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
                and center_vel is set. If only one of center_mass and center_vel are set,
                and force_origin is True, then the other is set to 0,0,0.
            seed: Random number seed, to create reproducible ICs. Optional.
    
         For example, here is how you might initialize a simulation that uses this:
          mysim = Simulation()
          mysim.add_IC(IC.TSIS(N=10000, maxrad=100*u.kpc, totmass=1e11*u.Msun))
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
        """Returns the positions, velocities, and masses for particles that
        form an isotropic Plummer model (BT eqs. 4.83 with n=5, 4.92, 2.44b).
        The parameters are:
            totmass: total mass (astropy Quantity)
            b: scale radius (astropy Quantity)
            N: number of particles
            center_pos: Force center of mass of simulation to here. Quantity array of size 3. Optional.
            center_vel: Force center of mass velocity of simulation to this. Quantity array
                of size 3. Optional.
            force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
                and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
                and center_vel is set. If only one of center_mass and center_vel are set,
                and force_origin is True, then the other is set to 0,0,0.
            seed: Random number seed, to create reproducible ICs. Optional.

        For example, here is how you might initialize a simulation that uses this:
         mysim = Simulation()
         mysim.add_IC(IC.Plummer(N=10000, b=1*u.pc, totmass=1e6*u.Msun))
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
    def Hernquist(N=None, a=None, totmass=None, center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Returns the positions, velocities, and masses for particles that
        form an isotropic Hernquist model (Hernquist 1990).
        The parameters are:
            totmass: total mass (astropy Quantity)
            a: scale radius (astropy Quantity)
            N: number of particles
            center_pos: Force center of mass of simulation to here. Quantity array of size 3. Optional.
            center_vel: Force center of mass velocity of simulation to this. Quantity array
                of size 3. Optional.
            force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
                and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
                and center_vel is set. If only one of center_mass and center_vel are set,
                and force_origin is True, then the other is set to 0,0,0.
            seed: Random number seed, to create reproducible ICs. Optional.
        
        For example, here is how you might initialize a simulation that uses this:
         mysim = Simulation()
         mysim.add_IC( IC.Hernquist(N=10000, a=1*u.kpc, totmass=1e10*u.Msun) )
         """
         
        rng = np.random.default_rng(seed)
     
        # Based on equation 10 from Hernquist 1990, I get r/a = (xi + sqrt(xi))/(1-xi).
        # get positions
        rad_xi = rng.uniform(0.0, 1.0, size=N)
        r_over_a = (rad_xi + np.sqrt(rad_xi))/(1. - rad_xi)
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
    def expdisk(sigma0=None, Rd=None, z0=None, sigmaR_Rd=None, halo_force=None, halo_force_args=None, N=None, \
        center_pos=None, center_vel=None, force_origin=True, seed=None):
        """Returns the positions, velocities, and masses for particles that
        form an exponential disk with a sech^2 vertical distribution that is
        in approximate equilibrium.
        The parameters are:
            sigma0: central surface density (astropy Quantity)
            Rd: exponential scale length (astropy Quantity)
            z0: scale height (astropy Quantity)
            sigmaR_Rd: radial velocity dispersion at R=Rd (astropy Quantity)
            halo_force: function that returns the force of any external potential,
                    or False if there isn't one. It is assumed that this is
                    spherically symmetric about the origin. You can write a wrapper
                    around Simulation.calculate_extra_acceleration() to include all
                    of the extra forces that have been added. Optional.
            halo_force_args: if halo_force is specified, this will be fed
                    into the force function as the second parameter. Optional.
            N: number of particles
            center_pos: Force center of mass of simulation to here. Quantity array of size 3. Optional.
            center_vel: Force center of mass velocity of simulation to this. Quantity array
                of size 3. Optional.
            force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
                and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
                and center_vel is set. If only one of center_mass and center_vel are set,
                and force_origin is True, then the other is set to 0,0,0.
            seed: Random number seed, to create reproducible ICs. Optional.

        For example, here is how you might initialize a simulation that uses this:
         mysim = Simulation()
         mysim.add_IC( IC.expdisk(N=10000, sigma0=200*u.Msun/u.pc**2, Rd=2*u.kpc,
                z0=0.5*u.kpc, sigmaR=10*u.km/u.s,
                halo_force=lambda x: (200*u.km/u.s)**2 / x) )
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
            # Input must be in kpc but with units stripped, because it's passed into derivative.
            y_R = rad/(2.*Rd_kpc)
            om2_disk = np.pi * const.G * sigma0 / Rd * (special.iv(0,y_R)*special.kv(0,y_R) -
                    special.iv(1,y_R)*special.kv(1,y_R))
            if halo_force is not None:
                xpos = rad * u.kpc
                ypos = np.zeros(shape=len(rad)) * u.kpc
                zpos = np.zeros(shape=len(rad)) * u.kpc
                pos = np.vstack((xpos,ypos,zpos)).T
                om2_halo = np.abs(halo_force(pos,halo_force_args)[:,0]) / rad
            else:
                om2_halo = 0. * om2_disk   # make sure units are correct
            return om2_disk + om2_halo

        Omega2 = om2(R.to(u.kpc).value)
        kappa2 = 4.*Omega2 + R * derivative(om2, R.to(u.kpc).value, 1e-3) / u.kpc

        sigma_R = sigmaR_Rd * np.exp(-R/Rd)
        sigma2_phi = sigma_R**2 * kappa2 / (4.*Omega2)
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
        center_pos: Force center of mass of simulation to here. Quantity array of size 3.
        center_vel: Force center of mass velocity of simulation to this. Quantity array
            of size 3.
        force_origin: Equivalent to setting center_pos=np.array([0,0,0])*u.kpc
            and center_vel=np.array([0,0,0])*u.km/u.s. Default True unless center_pos
            and center_vel is set. If only one of center_mass and center_vel are set,
            and force_origin is True, then the other is set to 0,0,0.
            
        Returns new (positions, velocities).
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


