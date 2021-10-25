from setuptools import setup, Extension
import numpy.distutils.misc_util

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
        name="gravhopper",
        version="1.0.0",
        author="Jeremy Bailin",
        author_email="jbailin@ua.edu",
        description="Simple N-body code for Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jbailinua/gravhopper",
        classifiers=[ \
            "Programming Language :: Python :: 3", \
            "License :: OSI Approved :: BSD License", \
            "Operating System :: OS Independent",
        ],
        package_dir={"":"."},
        packages=setuptools.find_packages(where="."),
        python_requires=">=3.6",
        
        ext_modules=[Extension("_jbgrav", ["_jbgrav.c"])],
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
        )


