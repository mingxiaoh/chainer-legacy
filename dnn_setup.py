from setuptools.extension import Extension
from numpy import get_include
from platform import system
import sys
import external

subdir = 'mkldnn'

# Sepcify prefix under which you put ipl_mkldnn
# prefix = '/usr/local'
mkldnn_root = external.mkldnn.root()
mkldnn_version = '472bbbf05ce5ff5c072811220c55cf9b5bbd96ad'


def prepare_mkldnn():
    external.mkldnn.prepare(mkldnn_version)


swig_opts = ['-c++', '-builtin', '-modern', '-modernargs',
             '-Idnn/py/mm', '-Idnn/py/primitives', '-Idnn/py/swig_utils',
             '-Idnn/include/primitives/', '-Idnn/include/mm/']

if sys.version_info.major < 3:
    swig_opts += ['-DNEWBUFFER_ON']

ccxx_opts = ['-std=c++11']
link_opts = ['-Wl,-z,now', '-Wl,-z,noexecstack',
             '-Wl,-rpath,' + mkldnn_root + '/lib', '-L' + mkldnn_root + '/lib']

includes = [get_include(), 'dnn/include', 'dnn/include/mkl', 'dnn/common', 'dnn/include/mm',
            'dnn/py/mm', 'dnn/py/primitives', 'dnn/include/primitives', 'dnn/include/primitives/ops', 'dnn/include/primitives/prim_mgr', mkldnn_root + '/include']
libraries = ['mkldnn', 'mklml_intel']

if system() == 'Linux':
    ccxx_opts += ['-fopenmp', '-DOPENMP_AFFINITY']
    libraries += ['boost_system', 'glog', 'm']
    src = ['dnn/py/dnn.i', 'dnn/mm/mem.cc', 'dnn/py/mm/mdarray.cc',
           'dnn/common/cpu_info.cc', 'dnn/common/utils.cc', 'dnn/common/common.cc',
           'dnn/primitives/ops/relu_fwd.cc', 'dnn/primitives/prim_mgr/relu_fwd_factory.cc',
           'dnn/primitives/relu.cc',
	       'dnn/primitives/ops/conv_fwd.cc', 'dnn/primitives/prim_mgr/conv_fwd_factory.cc',
           'dnn/primitives/ops/conv_bwd_weights.cc', 'dnn/primitives/prim_mgr/conv_bwd_weights_factory.cc',
           'dnn/primitives/ops/conv_bwd_data.cc', 'dnn/primitives/prim_mgr/conv_bwd_data_factory.cc',
           'dnn/primitives/ops/reorder_op.cc', 'dnn/primitives/prim_mgr/reorder_factory.cc',
           'dnn/primitives/conv.cc',
           'dnn/primitives/ops/pooling_fwd.cc', 'dnn/primitives/prim_mgr/pooling_fwd_factory.cc',
           'dnn/primitives/pooling.cc',
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

packages = ['dnn']
