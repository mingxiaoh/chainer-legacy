#!/usr/bin/env python

import pkg_resources

from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools import setup
from setuptools.extension import Extension

from numpy import get_include

setup_requires = []
install_requires = [
    'filelock',
    'nose',
    'numpy>=1.9.0',
    'protobuf',
    'six>=1.9.0',
]
cupy_require = 'cupy==1.0.0b1'

cupy_pkg = None
try:
    cupy_pkg = pkg_resources.get_distribution('cupy')
except pkg_resources.DistributionNotFound:
    pass

if cupy_pkg is not None:
    install_requires.append(cupy_require)
    print('Use %s' % cupy_require)

class _build_py(build_py):
    def run(self):
        self.run_command('build_ext')
        build_py.run(self)

class _install(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)

swig_opts=['-c++', '-Imkldnn', '-relativeimport', '-builtin', '-modern', '-modernargs']
ccxx_opts=['-std=c++11', '-O0', '-g']

ext_modules=[Extension("mkldnn._c_api", sources=['mkldnn/c_api.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._support", sources=['mkldnn/support.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._memory", sources=['mkldnn/memory.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._inner_product_forward", sources=['mkldnn/inner_product_forward.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._inner_product_backward_data", sources=['mkldnn/inner_product_backward_data.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._inner_product_backward_weights", sources=['mkldnn/inner_product_backward_weights.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._convolution_forward", sources=['mkldnn/convolution_forward.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._convolution_backward_data", sources=['mkldnn/convolution_backward_data.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._convolution_backward_weights", sources=['mkldnn/convolution_backward_weights.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._reorder", sources=['mkldnn/reorder.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._mdarray", sources=['mkldnn/mdarray.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, include_dirs=[get_include()], libraries=['mkldnn'])]

setup(
    name='chainer',
    version='2.0.0b1',
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
              'mkldnn'],
    ext_modules=ext_modules,
    cmdclass={'install':_install, 'build_py':_build_py},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
