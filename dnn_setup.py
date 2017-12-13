from setuptools.extension import Extension
from numpy import get_include
from platform import system
import sys
import external

subdir = 'mkldnn'

# Sepcify prefix under which you put ipl_mkldnn
# prefix = '/usr/local'
mkldnn_root = external.mkldnn.root()
mkldnn_version = 'ba482eca9459e3b9a8256ab07f9afa41dba34b9e'


def prepare_mkldnn():
    external.mkldnn.prepare(mkldnn_version)


swig_opts = ['-c++', '-builtin', '-modern', '-modernargs',
             '-Idnn/py/mm', '-Idnn/py/primitives', '-Idnn/py/swig_utils',
             '-Idnn/include/primitives/', '-Idnn/include/mm/']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11', '-Wno-unknown-pragmas']
link_opts = ['-Wl,-z,now', '-Wl,-z,noexecstack',
             '-Wl,-rpath,' + mkldnn_root + '/lib', '-L' + mkldnn_root + '/lib']

includes = [get_include(),
            'dnn/include',
            'dnn/include/mkl',
            'dnn/common',
            'dnn/include/mm',
            'dnn/py/mm',
            'dnn/py/primitives',
            'dnn/include/primitives',
            'dnn/include/blas',
            'dnn/include/primitives/ops',
            'dnn/include/primitives/prim_mgr',
            mkldnn_root + '/include']

libraries = ['mkldnn', 'mklml_intel']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['boost_system', 'glog', 'm']
    src = ['dnn/py/dnn.i',
           'dnn/mm/mem.cc',
           'dnn/mm/tensor.cc',
           'dnn/py/mm/mdarray.cc',
           'dnn/common/cpu_info.cc',
           'dnn/common/utils.cc',
           'dnn/common/common.cc',
           'dnn/blas/sum.cc',
           'dnn/py/mm/basic.cc',
           'dnn/primitives/ops/relu_fwd.cc',
           'dnn/primitives/ops/relu_bwd.cc',
           'dnn/primitives/relu.cc',
           'dnn/primitives/ops/conv_fwd.cc',
           'dnn/primitives/ops/conv_bwd_weights.cc',
           'dnn/primitives/ops/conv_bwd_data.cc',
           'dnn/primitives/ops/reorder_op.cc',
           'dnn/primitives/conv.cc',
           'dnn/primitives/ops/pooling_fwd.cc',
           'dnn/primitives/ops/pooling_bwd.cc',
           'dnn/primitives/pooling.cc',
           'dnn/primitives/ops/linear_fwd.cc',
           'dnn/primitives/ops/linear_bwd_weights.cc',
           'dnn/primitives/ops/linear_bwd_data.cc',
           'dnn/primitives/linear.cc',
           'dnn/primitives/bn.cc',
           'dnn/primitives/ops/bn_fwd.cc',
           'dnn/primitives/ops/bn_bwd.cc',
           'dnn/primitives/ops/concat_fwd.cc',
           'dnn/primitives/ops/concat_bwd.cc',
           'dnn/primitives/concat.cc',
           'dnn/primitives/ops/lrn_fwd.cc',
           'dnn/primitives/ops/lrn_bwd.cc',
           'dnn/primitives/lrn.cc',
           'dnn/primitives/dropout.cc'
           ]
else:
    # TODO
    src = ['mkldnn/mdarray.i', 'mkldnn/mdarray.cc']

ext_modules = []

ext = Extension(
    'dnn._dnn', sources=src,
    swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, extra_link_args=link_opts,
    include_dirs=includes, libraries=libraries)

ext_modules.append(ext)

packages = ['dnn', 'dnn.cosim']
