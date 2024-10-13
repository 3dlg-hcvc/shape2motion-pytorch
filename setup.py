#!/usr/bin/env python3
from setuptools import find_packages, setup

requirements = [
    "hydra-core==1.2.0",
    "urdfpy==0.0.22",
    "PyYAML==6.0",
    "pandas==1.1.5",
    "matplotlib==3.3.4",
    "trimesh==3.9.36",
    "progress==1.6",
    "tqdm==4.39.0",
    "dgl==0.6.0",
]

setup(
    name="shape2motion",
    version="1.0",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/shape2motion-pytorch",
    packages=find_packages(exclude=("configs", "tests")),
    include_package_data=True,
    install_requires=requirements,
)
