from setuptools import setup, Extension, find_packages
import numpy.distutils.misc_util

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
        name="gravhopper",
        version="1.1.0",
        author="Jeremy Bailin",
        author_email="jbailin@ua.edu",
        description="Simple N-body code for Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/jbailinua/gravhopper",
        license="BSD",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
        ],
        package_dir={"":"."},
        packages=find_packages(),
        include_package_data=True,
        
        python_requires=">=3.6",
        install_requires=[
            'numpy>=1.21',
            'scipy>=1.3',
            'matplotlib>=2.2',
            'astropy>=4.0',
        ],
        
        ext_modules=[Extension("gravhopper._jbgrav", ["gravhopper/_jbgrav.c"])],
        include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
        )

