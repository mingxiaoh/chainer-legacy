#!/usr/bin/env python

import pkg_resources

from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools import setup
# from setuptools.extension import Extension
import mkldnn_setup
# from numpy import get_include

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

packages = [
          'chainer',
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
          'chainer.utils']

ext_modules = mkldnn_setup.ext_modules
packages += mkldnn_setup.packages

setup(
    name='chainer',
    version='2.0.0b1',
    description='A flexible framework of neural networks',
    author='Seiya Tokui',
    author_email='tokui@preferred.jp',
    url='http://chainer.org/',
    license='MIT License',
    packages=packages,
    ext_modules=ext_modules,
    cmdclass={'install': _install, 'build_py': _build_py},
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=['mock',
                   'nose'],
)
