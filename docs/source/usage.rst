Basic Usage
===========

Usage of GravHopper generally involves the following steps:
 1. :ref:`simulation`
 2. :ref:`ICs`
 3. :ref:`external_forces`
 4. :ref:`run`
 5. :ref:`analyze`

See also these :ref:`Examples`.


.. _simulation:

Create a Simulation object
--------------------------

The fundamental unit of GravHopper is a :class:`~gravhopper.Simulation` object, which contains all of the
particle positions, velocities, and masses, at all timesteps available, and any external
forces that are being used.

To create a new :class:`~gravhopper.Simulation`::

    from gravhopper import Simulation
    sim = Simulation()
    
The fundamental parameters of a simulation that can be set during initialization are
the length of time step ``dt``, the softening length ``eps``, and the gravity calculation
``algorithm`` (these can all be changed after initialization using
:meth:`~gravhopper.Simulation.set_dt`, :meth:`~gravhopper.Simulation.set_eps` , and
:meth:`~gravhopper.Simulation.set_algorithm`). For example, to create a simulation with a 
time step of 5,000 years and a softening length of 0.05 pc::

    from astropy import units as u
    sim = Simulation(dt=5e3*u.yr, eps=0.05*u.pc)
    


.. _ICs:

Create initial conditions
-------------------------

Initial conditions consist of the positions, velocities, and masses of all of the
particles at the beginning of the simulation.

Initial conditions are added using a dict with the following key/value pairs:
    ``pos``: An (Np,3) array of positions
    
    ``vel``: An (Np,3) array of velocities
    
    ``mass``: An (Np) array of masses
    
Each of these must be an astropy Quantity with appropriate dimensions. If adding only
one particle, ``pos`` and ``vel`` can just have shape (3) but ``mass`` must still
must be an array with length 1 in that case.

Initial conditions are added to a simulation using :meth:`~gravhopper.Simulation.add_IC`::

    sim.add_IC( {'pos':[1,0,0]*u.pc, 'vel':[0,2,0]*u.km/u.s, 'mass':[1e3]*u.Msun} )

:meth:`~gravhopper.Simulation.add_IC` can be called multiple times for the same simulation to add more particles.

GravHopper contains a number of functions for generating a variety of useful ICs in the
:class:`~gravhopper.IC` namespace. For example, to create a Plummer sphere consisting of 2000 particles,
a scale length of 1 pc, and a total mass of one million solar masses::

    from gravhopper import IC
    Plummer_IC = IC.Plummer(N=2000, b=1*u.pc, totmass=1e6*u.Msun)
    sim.add_IC(Plummer_IC)
    
Among the other types of distributions that can be included are a Hernquist sphere,
an exponential disk, and a truncated singular isothermal sphere. It also contains
functions for creating initial conditions from a ``pynbody`` snapshot, or from a
``galpy`` distribution function object. All of these functions have the ability to place
the distribution at an arbitrary position and with an arbitrary global velocity using the
``center_pos`` and ``center_vel`` arguments. For example, to create a Hernquist sphere
at y=25 kpc with an initial velocity v\ :sub:`x`\ =100 km/s::

    Hernquist_IC = IC.Hernquist(N=2000, a=2*u.kpc, totmass=1e9*u.Msun, center_pos=[0,25,0]*u.kpc,
        center_vel=[100,0,0]*u.km/u.s))


.. _external_forces:

Add external forces (optional)
------------------------------

An external force field can be added to the simulation, so particles feel both the
N-body force from the particle distribution and the external force. Forces can be
implemented as simple functions, or using ``galpy`` or ``gala`` potential objects. A force
that only depends on position is added using :meth:`~gravhopper.Simulation.add_external_force`;
one that also depends on time is added using :meth:`~gravhopper.Simulation.add_external_timedependent_force`;
and one that depends on velocity (i.e. a dissipative force) is added using
:meth:`~gravhopper.Simulation.add_external_velocitydependent_force`.
Multiple external forces can be added by calling these functions multiple times.

For example, you could add a ``gala`` NFW potential with a scale mass of 10\ :sup:`11` 
M\ :sub:`sun` and a scale length of 20 kpc located at (x,y,z)=(0,0,50) kpc as::

    from gala.potential import NFWPotential
    from gala.units import galactic
    NFWpot = NFWPotential(m=1e11*u.Msun, r_s=20*u.kpc, units=galactic, origin=[0,0,50]*u.kpc)
    sim.add_external_force(NFWpot)



.. _run:

Run the simulation
------------------

Perform *N* steps of the simulation, each of length the current time step ``dt`` using
:meth:`~gravhopper.Simulation.run`::

    sim.run(N)
    
A simulation can be continued from where it left off by calling :meth:`~gravhopper.Simulation.run` again, possibly after changing
parameters or adding new external forces (but **not** changing the number/properties of
any the particles -- if you want to do that, create a new set of ICs based on the final
snapshot and perform a new simulation starting with those). This can be useful, for
example, to use longer timesteps initially to let a system come to equilibrium, then
use short timesteps so you can analyze the simulation with finer time resolution.



.. _analyze:

Analyze the output
------------------

The full positions and velocities of all particles at all timesteps are available via
the :attr:`~gravhopper.Simulation.positions` and :attr:`~gravhopper.Simulation.velocities` attributes, and the time of each snapshot is in
the :attr:`~gravhopper.Simulation.times` attribute. Each of 
:attr:`~gravhopper.Simulation.positions` and :attr:`~gravhopper.Simulation.velocities` is an astropy Quantity
array of shape (Nsnap, Np, 3). So, you could plot the x velocity of particle number 35
as a function of time using::

    import matplotlib.pyplot as plt
    plt.plot(sim.times, sim.velocities[:,35,0])
    plt.xlabel(f't ({sim.times.unit})')
    plt.ylabel(f'$v_x$ ({sim.velocities.unit})')
    
Or plot the x-y track of particle number 10::

    plt.subplot(111, aspect=1.0)
    plt.plot(sim.positions[:,10,0], sim.positions[:,10,1])
    plt.xlabel(f'x ({sim.positions.unit})')
    plt.ylabel(f'y ({sim.positions.unit})')
    
You can also use the built-in :meth:`~gravhopper.Simulation.plot_particles` method to look at 2D projections of
all of the particles at a particular point in time, either in position space or
velocity space. For example, to look at the x-z positions of all particles in the final
snapshot, with axes in units of pc::

    sim.plot_particles(coords='xz', snap='final', unit=u.pc)
    
Or to see the distribution of the x-y velocities of only particles 1000-1999 at
snapshot number 25::

    sim.plot_particles(parm='vel', snap=25, particle_range=[1000,2000])
    
These particle plots can be automatically turned into a movie of the simulation
evolving using the :meth:`~gravhopper.Simulation.movie_particles` method::

    sim.movie_particles('my-movie.mp4', unit=u.pc)
    
If you want to make movies of other aspects of the simulation, the :meth:`~gravhopper.Simulation.movie_particles`
source code provides a useful template.
    
``pynbody`` has a large number of routines that are useful for analyzing N-body simulation
outputs. You can take advantage of these by creating a ``SimSnap`` from any
snapshot of the simulation using the :meth:`~gravhopper.Simulation.pyn_snap` method.
For example, you could plot a before-and-after 3D density
profile using::

    from pynbody.analysis.profile import Profile
    s_IC = sim.pyn_snap(timestep=0)   # Note this always puts length in kpc
    s_final = sim.pyn_snap()
    p_IC = Profile(s_IC, ndim=3, min=0.0001, max=0.02, nbins=20)
    p_final = Profile(s_final, ndim=3, min=0.0001, max=0.02, nbins=20)
    plt.plot(p_IC['rbins'].in_units('pc'), p_IC['density'], label='initial')
    plt.plot(p_final['rbins'].in_units('pc'), p_final['density'], label='final')
    plt.yscale('log')
    plt.xlabel('r (pc)')
    plt.ylabel(f'$\\rho$ (${p_IC["density"].units.latex()}$)')
    plt.legend()
