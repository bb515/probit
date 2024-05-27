"""
Setup script for probit.

This setup is required or else
    >> ModuleNotFoundError: No module named 'probit'
will occur.
"""

from setuptools import setup, find_packages
import pathlib


# The directory containing this file
HERE = pathlib.Path(__file__).parent


# The text of the README file
README = (HERE / "README.md").read_text()


# The text of the README file
LICENSE = (HERE / "LICENSE.rst").read_text()


setup(
    name="probit",
    python_requires=">=3.8",
    description="probit is a simple and accessible Gaussian process implementation in Jax",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/probit",
    author="Benjamin Boys, Toby Boyne, Ieronymos Maxoutis",
    license="MIT",
    license_file=LICENSE,
    packages=find_packages(exclude=["*.test"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "jaxopt>=0.5.5",
        "backends>=1.4.32",
        "mlkernels>=0.3.6",
    ],
)
