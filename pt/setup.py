# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

from setuptools import setup, find_packages

setup(
    name="vmz",
    version="0.0.1",
    description="Video to text correspondence",
    author="",
    author_email="bruno@kor.bar",
    url="",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["torch", "torchvision", "submitit"],
    packages=find_packages(),
)

