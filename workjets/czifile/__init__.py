# -*- coding: utf-8 -*-
"""
    @ description：
        新增处理czi文件的代码  主要作为添加CPP动态库支持的样例
        并没有完全添加libczi（cpp）项目中的所有功能，如需要可以参考一下连接：
            https://github.com/zeiss-microscopy/libCZI
            https://github.com/elhuhdron/pylibczi/

        读取处理czi文件的   使用Python函数调用 由_czifile下 c/cpp写好的python接口 进一步封装

        TODO: 这里的czifile文件夹下的代码调用由 _czifile下c/cpp编译的pyd库
              而 _czifile下c/cpp 代码调用了C++编译好的库(dll/lib等) !!!
              通过setuptools.setup的data_files指定这些"数据文件"进行复制
    @ date:
    @ author: achange
"""

from .core import CziFile




if __name__ == '__main__':
    pass

