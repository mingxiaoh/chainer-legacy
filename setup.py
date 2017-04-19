#!/usr/bin/env python

import pkg_resources

from setuptools import setup
from setuptools.extension import Extension

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

swig_opts=['-c++', '-I/usr/local/include', '-relativeimport', '-builtin']
ccxx_opts=['-std=c++11', '-O0', '-g']

ext_modules=[Extension("mkldnn._c_api", sources=['mkldnn/c_api.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._support", sources=['mkldnn/support.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._memory", sources=['mkldnn/memory.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn']),
    Extension("mkldnn._mdarray", sources=['mkldnn/mdarray.i'], swig_opts=swig_opts,
    extra_compile_args=ccxx_opts, libraries=['mkldnn'])]

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
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
