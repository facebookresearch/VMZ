#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup


pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    'numpy',
    'six',
    pytorch_dep,
]


setup(
    name="vmz",
    version="0.1",
    author="FAIR",
    author_email='bruno@kor.bar',
    url="unknown",
    description="Video Model Zoo",
    install_requires=requirements,
    extras_require={
        "scipy": ["scipy"],
    },
    packages=find_packages()),
)