#!/usr/bin/env python

from distutils.command.build_py import build_py
from setuptools.command.install import install
from setuptools import setup
from setuptools.extension import Extension
import numpy

setup_requires = []
install_requires = [
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
    'glog',
]


class _build_py(build_py):
    def run(self):
        self.run_command('build_ext')
        build_py.run(self)


class _install(install):
    def install(self):
        print("run home made")
        self.run_command('build_ext')
        install.run(self)


extensions = [
    Extension(
        "mkldpy._mkldnn",
        sources=[
                "mkldpy/relu4d.cc",
                "mkldpy/relu.cc",
                "mkldpy/conv.cc",
                "mkldpy/deconv.cc",
                "mkldpy/concat.cc",
                "mkldpy/common.cc",
                "mkldpy/cpu_info.cc",
                "mkldpy/layer_factory.cc",
                "mkldpy/linear.cc",
                "mkldpy/lrn.cc",
                "mkldpy/pooling.cc",
                "mkldpy/max_pooling.cc",
                "mkldpy/avg_pooling.cc",
                "mkldpy/softmax.cc",
                "mkldpy/softmax_cross_entropy.cc",
                "mkldpy/sum.cc",
                "mkldpy/utils.cc",
                "mkldpy/batch_normalization.cc",
                "mkldpy/mkldnn.i"
                ],
        swig_opts=["-c++"],
        extra_compile_args=["-std=c++11", "-fopenmp"],
        include_dirs=["mkldpy/incl/", numpy.get_include()],
        libraries=['glog', 'stdc++', 'boost_system', 'mkldnn', 'm'],
    )
]

setup(
    name='chainer',
    version='2.0.0a1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    license='MIT License',
    packages=['chainer',
              'chainer.dataset',
              'chainer.datasets',
              'chainer.functions',
              'chainer.functions.activation',
              'chainer.functions.array',
              'chainer.functions.caffe',
              'chainer.functions.connection',
              'chainer.functions.evaluation',
              'chainer.functions.loss',
              'chainer.functions.math',
              'chainer.functions.noise',
              'chainer.functions.normalization',
              'chainer.functions.pooling',
              'chainer.functions.theano',
              'chainer.functions.util',
              'chainer.function_hooks',
              'chainer.iterators',
              'chainer.initializers',
              'chainer.links',
              'chainer.links.activation',
              'chainer.links.caffe',
              'chainer.links.caffe.protobuf2',
              'chainer.links.caffe.protobuf3',
              'chainer.links.connection',
              'chainer.links.loss',
              'chainer.links.model',
              'chainer.links.model.vision',
              'chainer.links.normalization',
              'chainer.links.theano',
              'chainer.optimizers',
              'chainer.serializers',
              'chainer.testing',
              'chainer.training',
              'chainer.training.extensions',
              'chainer.training.triggers',
              'chainer.utils',
              'mkldpy',
              ],
    ext_modules=extensions,
    cmdclass={'build_py': _build_py, 'install': _install},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
