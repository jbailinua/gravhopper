.. GravHopper documentation master file, created by
   sphinx-quickstart on Sun Dec 12 14:06:50 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GravHopper's documentation
=====================================

*"They told me computers could only do arithmetic." -- Grace Hopper*

`GravHopper <https://github.com/jbailinua/gravhopper>`_ is a gravitational N-body simulation code written by Jeremy Bailin, named in
honor both of pioneering computer scientist Grace Hopper, and the leapfrog integration
algorithm.

GravHopper combines a simple Python interface for ease of use with a C backend for speed.
The code is designed to be easy to understand and use for teaching in advanced
undergraduate and graduate courses, while being efficient and scalable enough to 
quickly run simulations with reasonable numbers of particles -- anything that doesn't
require cluster-scale computing is doable with GravHopper. It includes the ability to
add external forces, and interfaces well with the galpy, gala, and pynbody packages.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation.rst
   usage.rst
   examples.rst
   reference.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
