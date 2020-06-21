# -*- coding: utf-8 -*-
"""
    @ description：

    @ date:
    @ author: geekac
"""

import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# todo: 项目依赖包
install_requires = [
    'medpy>=0.4.0',
    'torch>=1.0.1',
    'numpy',

]



setuptools.setup(
    name="workjets",
    version="0.1.2",
    author="geekac",
    author_email="geekac@163.com",
    description="A common using tools library for work efficiently.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geekac/workjets",
    packages=setuptools.find_packages(),
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    install_requires=install_requires,
    license='MIT',

)
