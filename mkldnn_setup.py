from setuptools.extension import Extension
from numpy import get_include

subdir = 'mkldnn'

modules = {
        'mkldnn.api._c_api' :
        ['mkldnn/api/c_api.i'],

        'mkldnn.api._support' :
        ['mkldnn/api/support.i'],

        'mkldnn.api._memory' :
        ['mkldnn/api/memory.i'],

        'mkldnn.api._inner_product_forward' :
        ['mkldnn/api/inner_product_forward.i'],

        'mkldnn.api._inner_product_backward_data' :
        ['mkldnn/api/inner_product_backward_data.i'],

        'mkldnn.api._inner_product_backward_weights' :
        ['mkldnn/api/inner_product_backward_weights.i'],

        'mkldnn.api._convolution_forward' :
        ['mkldnn/api/convolution_forward.i'],

        'mkldnn.api._convolution_backward_data' :
        ['mkldnn/api/convolution_backward_data.i'],

        'mkldnn.api._convolution_backward_weights' :
        ['mkldnn/api/convolution_backward_weights.i'],

        'mkldnn.api._relu_forward' :
        ['mkldnn/api/relu_forward.i'],

        'mkldnn.api._relu_backward' :
        ['mkldnn/api/relu_backward.i'],

        'mkldnn.api._pooling_forward' :
        ['mkldnn/api/pooling_forward.i'],

        'mkldnn.api._pooling_backward' :
        ['mkldnn/api/pooling_backward.i'],

        'mkldnn.api._lrn_forward' :
        ['mkldnn/api/lrn_forward.i'],

        'mkldnn.api._lrn_backward' :
        ['mkldnn/api/lrn_backward.i'],

        'mkldnn.api._reorder' : ['mkldnn/api/reorder.i'],

        'mkldnn._mdarray' : ['mkldnn/mdarray.i']}

swig_opts=['-c++', '-Imkldnn', '-relativeimport',
        '-builtin', '-modern', '-modernargs',
        '-Imkldnn/api', '-Imkldnn', '-Imkldnn/swig_utils']

ccxx_opts=['-std=c++11', '-O0', '-g']

includes = [get_include(), 'mkldnn', 'mkldnn/swig_utils']
libraries = ['mkldnn']

ext_modules = []
for m, s in modules.items():
    ext = Extension(m, sources=s,
            swig_opts=swig_opts,
            extra_compile_args=ccxx_opts, include_dirs=includes, libraries=libraries)

    ext_modules.append(ext)

packages = ['mkldnn', 'mkldnn.api', 'mkldnn.chainer']
