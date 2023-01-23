"""
Setup script for probit.

This setup is required or else
    >> ModuleNotFoundError: No module named 'probit'
will occur.
"""
from setuptools import setup, find_packages
import subprocess
import pathlib


version = subprocess.run(['git', 'describe', '--tags'], stdout=subprocess.PIPE).stdout.decod("utf-8").strip()

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

extra_compile_args = ['-O3']
extra_link_args = []


setup(
    name="probit",
    version=version,
    python_requires=">=3.6",
    description="A simple and accessible Gaussian process package in Jax.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/probit",
    author="Benjamin Boys, Toby Boyne, Ieronymos Maxoutis",
    license="MIT",
    license_file=LICENSE.rst,
    packages=find_packages(exclude=['*.test']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'jaxlib>=0.4.1',
        'jax>=0.4.1',
        'jaxopt>=0.5.5'
        'backends>=1.4.32',
        'mlkernels>=0.3.6',
        ]
    )
