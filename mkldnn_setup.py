from setuptools.extension import Extension
from numpy import get_include
from platform import system
import sys
# import os

subdir = 'mkldnn'

# Sepcify prefix under which you put ipl_mkldnn
prefix = '/usr/local'

modules = {
        'mkldnn.api._c_api':
        ['mkldnn/api/c_api.i'],

        'mkldnn.api._support':
        ['mkldnn/api/support.i'],

        'mkldnn.api._memory':
        ['mkldnn/api/memory.i', 'mkldnn/utils.cc'],

        'mkldnn.api._inner_product_forward':
        ['mkldnn/api/inner_product_forward.i'],

        'mkldnn.api._inner_product_backward_data':
        ['mkldnn/api/inner_product_backward_data.i'],

        'mkldnn.api._inner_product_backward_weights':
        ['mkldnn/api/inner_product_backward_weights.i'],

        'mkldnn.api._convolution_forward':
        ['mkldnn/api/convolution_forward.i'],

        'mkldnn.api._convolution_backward_data':
        ['mkldnn/api/convolution_backward_data.i'],

        'mkldnn.api._convolution_backward_weights':
        ['mkldnn/api/convolution_backward_weights.i'],

        'mkldnn.api._eltwise_forward':
        ['mkldnn/api/eltwise_forward.i'],

        'mkldnn.api._eltwise_backward':
        ['mkldnn/api/eltwise_backward.i'],

        'mkldnn.api._pooling_forward':
        ['mkldnn/api/pooling_forward.i'],

        'mkldnn.api._pooling_backward':
        ['mkldnn/api/pooling_backward.i'],

        'mkldnn.api._lrn_forward':
        ['mkldnn/api/lrn_forward.i'],

        'mkldnn.api._lrn_backward':
        ['mkldnn/api/lrn_backward.i'],

        'mkldnn.api._sum': ['mkldnn/api/sum.i'],
        'mkldnn.api._reorder': ['mkldnn/api/reorder.i'],
        'mkldnn.api._concat': ['mkldnn/api/concat.i'],
        'mkldnn.api._view': ['mkldnn/api/view.i'],

        'mkldnn.api._bn_forward': ['mkldnn/api/bn_forward.i'],
        'mkldnn.api._bn_backward': ['mkldnn/api/bn_backward.i']}


swig_opts = [
            '-c++', '-Imkldnn', '-relativeimport',
            '-builtin', '-modern', '-modernargs',
            '-Imkldnn/api', '-Imkldnn', '-Imkldnn/swig_utils']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11']
link_opts = ['-Wl,-z,now', '-Wl,-z,noexecstack', '-L' + prefix + '/lib']

includes = [get_include(), 'mkldnn', 'mkldnn/swig_utils', prefix + '/include']
libraries = ['mkldnn']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['boost_system', 'glog', 'm']
    mdarray_src = ['mkldnn/mdarray.i', 'mkldnn/mdarray.cc', 'mkldnn/cpu_info.cc']
else:
    mdarray_src = ['mkldnn/mdarray.i', 'mkldnn/mdarray.cc']

ext_modules = []
for m, s in modules.items():
    ext = Extension(
        m, sources=s,
        swig_opts=swig_opts,
        extra_compile_args=ccxx_opts, extra_link_args=link_opts,
        include_dirs=includes, libraries=libraries)

    ext_modules.append(ext)

ext = Extension(
        'mkldnn._mdarray', sources=mdarray_src,
        swig_opts=swig_opts,
        extra_compile_args=ccxx_opts, extra_link_args=link_opts,
        include_dirs=includes, libraries=libraries)

ext_modules.append(ext)

packages = ['mkldnn', 'mkldnn.api', 'mkldnn.chainer']
