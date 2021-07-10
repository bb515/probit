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

extra_compile_args = ['-O3']
extra_link_args = []

arguments = [
    "abalone",
    "Auto-Mpg",
    "bostonhousing",
    "Diabetes",
    "machinecpu",
    "pyrimidines",
    "stocksdomain",
    "triazines",
    "wisconsin",
    "false"
]
data_list = []
for argument in arguments:
    data_list.append('data/{}/*.npz'.format(argument))
    data_list.append('data/{}/5bins/*.npz'.format(argument))
    data_list.append('data/{}/10bins/*.npz'.format(argument))


setup(
    name="probit",
    version="0.1.0",
    description="A fast ordinal regression with GP priors package for python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bb515/VariationalBayesianMultinomialProbitRegressionwithGaussianProcessPriors",
    author="Ben Boys",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    # package_dir={'mypkg': 'src/mypkg'},
    package_data={'probit': data_list},
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'h5py',
        ]
    )