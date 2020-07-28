# -*- coding: utf-8 -*-
"""
    @ description：
        自己写的pypi库代码
        TODO: 注意啊  numpy构建whl的代码大部分重写了setuptools相关功能，以后有需求再更深入了解！
    @ date:
    @ author: geekac
"""
import setuptools
from extension_czi import build_libczi, ext_module_czi, SpecializedClean, SpecializedBuildExt


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()



# todo: 项目依赖包
install_requires = [
    'medpy>=0.4.0',
    'torch>=1.0.1',
    'numpy',
]


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

# todo: 执行代码 构建czifile扩展
module_czi = ext_module_czi()


setuptools.setup(
    name="workjets",
    version="0.1.4",
    author="geekac",
    author_email="geekac@163.com",
    description="A common using tools library for work efficiently.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geekac/workjets",

    # todo: Add for extension czifile:
    ext_modules=[module_czi],
    cmdclass={
        'build_ext': SpecializedBuildExt,
        'clean': SpecializedClean,
    },

    # data_files = data_files,      # todo: 如果直接使用编译好的库 可以写在这里！ list形式
    #  [('', ['D:\\achange\\code\\Github\\pylibczi-master\\libCZI\\build\\Src\\libCZI\\Release\\libCZI.dll'])]

    packages=setuptools.find_packages(),
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    # platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    install_requires=install_requires,
    license='MIT',
)
