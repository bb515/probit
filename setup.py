"""
Setup script for probit_jax.

This setup is required or else
    >> ModuleNotFoundError: No module named 'probit_jax'
will occur.
"""
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

extra_compile_args = ['-O3']
extra_link_args = []


setup(
    name="probit_jax",
    version="0.1.0",
    description="A classification with GP priors package for python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/probit_jax",
    author="Benjamin Boys, Toby Boyne, Ieronymos Maxoutis",
    license="MIT",
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
