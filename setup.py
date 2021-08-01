"""
Install using pip, e.g. pip install ./FDC_bias_correction
use --no-deps to prevent re-installation of dependencies
use -I to force re-install
"""

from setuptools import setup, find_packages
from glob import glob

scripts = glob("statistical_transformations/scripts/*.py")

setup(
    name="statistical_transformations",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/dlagrava/fdc_bias_correction",
    description="Statistical transformations for flow",
    install_requires=[],
    scripts=scripts
)
