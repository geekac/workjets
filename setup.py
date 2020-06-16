# -*- coding: utf-8 -*-
"""
    @ description：

    @ date:
    @ author: geekac
"""

import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="workjets",
    version="0.1.0",
    author="geekac",
    author_email="geekac@163.com",
    description="A common using tools library for work efficiently.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geekac/workjets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
