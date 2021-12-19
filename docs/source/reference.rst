Reference
=========

GravHopper consists of the following submodules:

* :ref:`simulation_object`
* :ref:`ic`
* :ref:`jbgrav`
* :ref:`exceptions`


.. _simulation_object:

Simulation
----------

General usage::

    from gravhopper import Simulation
    sim = Simulation(dt=<timestep>, eps=<softening>)
    sim.add_IC(<initial conditions>)
    sim.run(<N>)
    sim.plot_particles()
    # Other analysis using sim.positions, sim.velocities, sim.times


.. autoclass:: gravhopper.Simulation
    :members: __init__, add_IC, add_external_force, add_external_timedependent_force,
        add_external_velocitydependent_force, run, pyn_snap, plot_particles, movie_particles,
        set_dt, get_dt, set_eps, get_eps, set_algorithm, get_algorithm, reset, snap, current_snap,
        prev_snap, calculate_acceleration
    


.. _ic:

IC
--

General usage::

    from gravhopper import IC
    IC_array = IC.Plummer(totmass=<mass>, a=<length>, N=<N>)
    sim.add_IC(IC_array)

Because random sampling is not guaranteed to have the center of mass or mean velocity
of exactly zero, all functions that involve random sampling take an argument
``force_origin`` (default: True) that shifts the final particle positions and velocities
to be at the origin, or to a pre-set other location or velocity using the ``center_pos``
and ``center_vel`` arguments.

All functions where GravHopper does the random sampling
allow a random seed to be set using the ``seed`` argument to facilitate reproducibility.
This seed is fed directly into ``numpy.random.default_rng()``.


.. autoclass:: gravhopper.IC
    :members:



.. _jbgrav:

jbgrav
------

The jbgrav module contains the functions that calculate the N-body force within a
simulation. They are used internally within :ref:`simulation_object`, but can also
be imported and called on their own::

    from gravhopper import jbgrav
    from astropy import units as u
    from numpy import np
    sun = {'pos':[0,0,0]*u.au, 'vel':[0,0,0]*u.km/u.s, 'mass':[1]*u.Msun}
    earth = {'pos':[1,0,0]*u.au, 'vel':[0,29.8,0]*u.km/u.s, 'mass':[1]*u.Mearth}
    solarsystem = {'pos':np.vstack((sun['pos'],earth['pos'])),
        'vel':np.vstack((sun['vel'],earth['vel'])),
        'mass':np.vstack((sun['mass'],earth['mass']))}
    accelerations = jbgrav.direct_summation(solarsystem, 0.01*u.au)

.. automodule:: gravhopper.jbgrav
    :members:



.. _exceptions:

Exceptions
----------

.. autoexception:: gravhopper.gravhopper.GravHopperException
.. autoexception:: gravhopper.gravhopper.UninitializedSimulationException
.. autoexception:: gravhopper.gravhopper.ICException
.. autoexception:: gravhopper.gravhopper.UnknownAlgorithmException
.. autoexception:: gravhopper.gravhopper.ExternalPackageException


