# -*- coding: utf-8 -*-
"""
    @ description：
        TODO: 拓展读取czi文件

        向库中添加需要调用第三方库 并使用cpp调用 python作为接口的扩展模块  ---
        python调用cmd命令行程序 cmake编译开源项目 --> 动态链接库

        (0) 平台怎么处理呢？

        (1) 添加CPP实现的功能集成到库中调用：   --- 这就涉及到不同平台不同的编译 --(0)
            这里使用的libCZI库为第三方库 可以编译为C++的dll和静态链接库
                此处选择调用cmake编译libCZI动态链接库  ---  方便包源码统一管理
                当然也可以自己编译好放在指定的位置！   ---  需要为不同平台编译好不同的动态链接库

            然后编写cpp代码调用生成的动态链接库  需要#include "Python.h"头文件等  相关具体细节此处不详述  见相关专题。

            1. 当自己写的库中有CPP实现的功能时，需要 `from setuptools import Extension` 扩展c files.


    @ date:
    @ author: achange
"""
import os
import sys
import numpy        # 扩展模块使用到了Numpy库
import shutil
import platform
import sysconfig
import subprocess

from glob import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext
# from distutils.command import clean
from distutils.command.clean import clean


EXTENSION_MODEL_NAME = '_czifile'


# 获取平台以设置构建参数
platform_ = platform.system()
architecture = platform.architecture()
win_arch = 'x64' if architecture[0] == '64bit' else 'x86'

# todo: 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__))
libczi_dir = os.path.join(script_dir, 'libCZI')
build_temp = os.path.join(libczi_dir, 'build')  # 构建结果存放目录 ./libCZI/build
include_libCZI = os.path.join(libczi_dir, 'Src')
lib_libCZI = os.path.join(build_temp, 'Src', 'libCZI')
lib_JxrDecode = os.path.join(build_temp, 'Src', 'JxrDecode')


# to statically link libCZI  静态链接库?
build_static = (platform_ != 'Windows')     # todo: False
if build_static:
    # xxx - does not work, not sure why
    #   copy libCZI dll using data_files instead
    libCZI_win_release = 'static\\ Release'
else:
    libCZI_win_release = 'Release'
    lib_libCZI = os.path.join(lib_libCZI, libCZI_win_release)
    lib_JxrDecode = os.path.join(lib_JxrDecode, libCZI_win_release)


# todo:  该函数是用于编译由c/cpp编写的第三方库libCZI !!
def build_libczi():

    env = os.environ.copy()
    cmake_args = []
    build_args = []

    if platform_ == 'Windows':
        cmake_args += ['-DCMAKE_GENERATOR_PLATFORM=' + win_arch]
        build_args += ['--config', libCZI_win_release]
    else:
        cmake_args += ['-DCMAKE_BUILD_TYPE:STRING=Release']
    if not os.path.exists(build_temp):
        os.makedirs(build_temp)

    def run_cmake(cmake_exe):   # 执行cmake命令
        config_cmd_list = [cmake_exe, libczi_dir] + cmake_args      # 配置项目参数(平台等)
        build_cmd_list = [cmake_exe, '--build', '.'] + build_args
        if platform_ == 'Windows':
            print(subprocess.list2cmdline(config_cmd_list))
            # cmake D:\achange\code\Github\pylibczi-master\libCZI -DCMAKE_GENERATOR_PLATFORM=x64
            print(subprocess.list2cmdline(build_cmd_list))
            # cmake --build . --config Release

        subprocess.check_call(config_cmd_list, cwd=build_temp, env=env)
        subprocess.check_call(build_cmd_list, cwd=build_temp, env=env)

    # todo: 首先尝试使用 pip安装的cmake  如果未安装 那就从系统环境寻找
    try:
        # try to use the pip installed cmake first  先找c:/programdata/miniconda3/cmake.exe进行编译
        run_cmake(os.path.join(os.path.dirname(sys.executable), 'cmake'))
    except OSError:
        # xxx - anaconda windows pip install cmake goes into a Scripts dir
        #   better option here?
        run_cmake('cmake')


def _get_extra_link_args():
    """ 编译时的链接参数
    """
    def safe_get_env_var_list(var):
        vlist = sysconfig.get_config_var(var)
        return [] if vlist is None else vlist.split()

    # platform specific compiler / linker options
    extra_compile_args = safe_get_env_var_list('CFLAGS')
    extra_link_args = safe_get_env_var_list('LDFLAGS')
    if platform_ == 'Windows':
        extra_compile_args += safe_get_env_var_list('CL')
        extra_compile_args += safe_get_env_var_list('_CL_')
        extra_link_args += safe_get_env_var_list('LINK')
        extra_link_args += safe_get_env_var_list('_LINK_')
        extra_compile_args += ['/Ox']
    else:
        extra_compile_args += ["-std=c++11", "-Wall", "-O3"]
        if platform_ == 'Linux':
            extra_compile_args += ["-fPIC"]
            if build_static:
                # need to link with g++ linker for static libstdc++ to work
                os.environ["LDSHARED"] = os.environ["CXX"] if 'CXX' in os.environ else 'g++'
                extra_link_args += ['-static-libstdc++', '-shared']
                # extra_link_args += ["-Wl,--no-undefined"] # will not work with manylinux
        elif platform_ == 'Darwin':
            mac_ver = platform.mac_ver()[0]  # xxx - how to know min mac version?
            extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=" + mac_ver]
        extra_link_args += extra_compile_args

    return extra_link_args, extra_compile_args


# todo:  该函数是用于编译由c/cpp编写的库功能  编译为pyd可以直接import !!
def ext_module_czi():
    # 扩展czi模块时候的参数：
    include_dirs = [numpy.get_include(), include_libCZI]
    static_libraries = []
    static_lib_dirs = []
    libraries = []
    library_dirs = []
    extra_objects = []

    # todo: 使用拓展python的C/C++源码  用于调用编译好的dll库 或者直接在cpp中实现库的功能
    sources = ['_czifile.cpp']

    extra_link_args, extra_compile_args = _get_extra_link_args()

    if build_static:
        static_libraries += ['libCZIStatic', 'JxrDecodeStatic']
        static_lib_dirs += [lib_libCZI, lib_JxrDecode]
    else:
        libraries += ['libCZI']
        library_dirs += [lib_libCZI]

    # second answer at
    # https://stackoverflow.com/questions/4597228/how-to-statically-link-a-library-when-compiling-a-python-module-extension
    if platform_ == 'Windows':
        libraries.extend(static_libraries)
        library_dirs.extend(static_lib_dirs)
    else:  # POSIX
        extra_objects += ['{}/lib{}.a'.format(d, l) for d, l in zip(static_lib_dirs, static_libraries)]
        include_dirs.append('/usr/local/include')
        library_dirs.append('/usr/local/lib')

    module_czi = Extension(
        EXTENSION_MODEL_NAME,
        define_macros=[('CZIFILE_VERSION', "0.1.4"), ],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        sources=[os.path.join(EXTENSION_MODEL_NAME, x) for x in sources],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++11',
        extra_objects=extra_objects,
    )

    return module_czi


# stackoverflow.com/questions/41169711/python-setuptools-distutils-custom-build-for-the-extra-package-with-makefile
class SpecializedBuildExt(build_ext):
    """Subclass of build_ext subcommand to build libCZI.
    """
    special_extension = EXTENSION_MODEL_NAME

    def build_extension(self, ext):
        if ext.name == self.special_extension:
            build_libczi()      # todo: build 3-party libczi !
            if not build_static:
                # xxx - hack to copy the dlls in case static build is not working (windows)
                # fixme dll复制！  前面那么多代码都是为了这个dll
                data_files = [('', glob(os.path.join(lib_libCZI, '*.dll')))]
                if self.distribution.data_files is None:
                    self.distribution.data_files = data_files
                else:
                    self.distribution.data_files += data_files
        # Build the c library's python interface with the parent build_extension method
        super(SpecializedBuildExt, self).build_extension(ext)


class SpecializedClean(clean):
    """Subclass of clean subcommand to clean libCZI.
    """
    def run(self, *args, **kwargs):
        print('Remove libCZI build dir')
        shutil.rmtree(build_temp, ignore_errors=True)
        clean.run(self, *args, **kwargs)



if __name__ == '__main__':
    build_libczi()

