import sys
from os import path

from setuptools import find_packages, setup

min_py_version = (3, 7)

if sys.version_info < min_py_version:
    sys.exit(
        "Project is only supported for Python {}.{} or higher".format(*min_py_version)
    )

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()


setup(
    name="src",
    version="0.0.1",
    description="Model for decision making",
    long_description="",
    author="Vaibhav Thakur (vaibrainium)",
    keywords="database organization",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        requirements,
        "numpy~=1.23.0",
        "scipy~=1.8.1",
        "scikit-learn~=0.24.2",
        "GPUtil~=1.4.0",
        "cupy-cuda102~=10.1.0",
        "seaborn~=0.11.2",
    ],
)
