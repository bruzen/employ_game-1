#!/usr/bin/env python
import imp
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'employ_game', 'version.py'))
description = "Prototype of employment game"
with open(os.path.join(root, 'README.md')) as readme:
    long_description = readme.read()

setup(
    name="employ_game",
    version=version_module.version,
    author="Terry Stewart and Kirsten Robinson",
    author_email="tcstewar@uwaterloo.ca",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'employ_game = employ_game:main',
        ]
    },
    scripts=[],
    description=description,
    long_description=long_description,
    requires=[
    ],
)
