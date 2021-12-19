.. _installation:

Installation
============

Dependencies
------------
 * `Astropy <https://www.astropy.org/>`_
 * NumPy, SciPy, Matplotlib
 * To use the `galpy <https://docs.galpy.org/>`_, `gala <https://github.com/adrn/gala>`_, or `pynbody <https://pynbody.github.io/pynbody/>`_ interface functions, they must be installed.
 * Saving movies requires `ffmpeg <https://www.ffmpeg.org/>`_

Install with pip (recommended)
------------------------------
The latest stable release can be installed using::

    pip install gravhopper
    
Binary wheels should be available for most systems. If one is not available for your
operating system, you will need a C compiler installed.


Install from github
-------------------

Use this if you want the current code and/or if you don't want to install it into the python path but just have a local version.

1. Clone or download the git repository::

    git clone https://github.com/jbailinua/gravhopper.git
    
2. Go into the gravhopper directory and build the code::

    cd gravhopper
    python setup.py build_ext --inplace
    
3. Copy the gravhopper subdirectory to wherever you want to use it.
