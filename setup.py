from setuptools import find_packages, setup

setup(
    name='src',
    author='Vaibhav Thakur (vaibrainium)',
    packages=find_packages(),
    install_requires=[
    "numpy~=1.23.0",
    "scipy~=1.8.1",
    "scikitlearn~=0.24.2",
    "GPUtil~=1.4.0",
    "cupy-cuda102~=10.1.0"
    "seaborn~=0.11.2"
    ],
)