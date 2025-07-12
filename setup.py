from setuptools import setup, Extension, find_packages
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
        name="gravhopper",
        version="1.2.0",
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
        
        python_requires=">=3.7",
        install_requires=[
            'numpy>=1.21',
            'scipy>=1.10',
            'matplotlib>=2.2',
            'astropy>=4.0',
        ],
        
        ext_modules=[Extension("gravhopper._jbgrav", ["gravhopper/_jbgrav.c"])],
        include_dirs=numpy.get_include(),
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")]
        )

