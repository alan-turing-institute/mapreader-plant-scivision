#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="mapreader_model",
    version="0.0.1",
    description="scivision test plugin, using mapreader",
    author="",
    author_email="",
    url="https://github.com/alan-turing-institute/mapreader-plant-scivision",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.7",
)
