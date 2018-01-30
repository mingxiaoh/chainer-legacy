<div align="center"><img src="docs/image/chainer_red_h.png" width="400"/></div>

# IntelChainer: Optimized-Chainer for Intel Architectures

[![GitHub license](https://img.shields.io/github/license/intel/chainer.svg)](https://github.com/intel/chainer)
[![travis](https://img.shields.io/travis/intel/chainer/master.svg)](https://travis-ci.org/intel/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)


Chainer* is a Python*-based deep learning framework aiming at flexibility and intuition. It provides automatic differentiation APIs based on the define-by-run approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. It supports various network architectures including feed-forward nets, convnets, recurrent nets and recursive nets. It also supports per-batch architectures. Forward computation can include any control flow statements of Python without lacking the ability of backpropagation. It makes code intuitive and easy to debug. Intel® optimization for Chainer, is currently integrated with the latest release of Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) 2017 optimized for Intel® Advanced Vector Extensions 2 (Intel® AVX) and Intel® Advanced Vector Extensions 512 (Intel®AVX-512) instructions which are supported in Intel® Xeon® and Intel® Xeon Phi™ processors.

## Recommended Environments
We recommend these Linux distributions.
- Ubuntu 14.04/16.04 LTS 64bit
- CentOS 7 64bit

The following versions of Python can be used: 
- 2.7.5+, 3.5.2+, and 3.6.0+

Above recommended environments are tested. We cannot guarantee that Intel® optimization for Chainer works on other environments including Windows* and macOS*, even if Intel optimization for Chainer looks to be running correctly.

## Dependencies
Before installing Chainer, we recommend to upgrade setuptools if you are using an old one:

$ pip install -U setuptools

The following packages are required to install Chainer.
- NumPy 1.9, 1.10, 1.11, 1.12, 1.13
- Six 1.9+
- Swig 3.0.9+

The following packages are optional dependencies. Chainer can be installed without them, in which case the corresponding features are not available.

Caffe model support
- protobuf 3.0+

Image dataset support
- pillow 2.3+

HDF5 serialization support
- h5py 2.5+

Testing utilities
- pytest 3.2.5+

Intel® MKL-DNN
- You don’t need to manually install Intel MKL-DNN, when build Intel optimization for Chainer, Intel MKL-DNN will be downloaded and built automatically, thus, boost, glog and gflags are also required.


## Install Chainer from source
You can use setup.py to install Chainer from the tarball:

```sh
$ python setup.py install
```

Use pip to uninstall Chainer:

```sh
$ pip uninstall chainer
```


## Run with Docker

We are providing the Docker image and Dockerfile for Ubuntu and Centos based on python2 and python3, respectively. For details see: [How to build and run Intel optimization for Chainer Docker image](https://github.com/intel/chainer/wiki/How-to-build-and-run-Intel-Chainer-Docker-image).



## Training Examples

Training test with mnist dataset:
```sh
$ cd examples/mnist
$ python train_mnist.py -g -1
```

Training test with cifar datasets:
- run the CIFAR-100 dataset:
```sh
$ cd examples/cifar
$ python train_cifar.py –g -1 --dataset='cifar100'
```
- run the CIFAR-10 dataset:
```sh
$ cd examples/cifar
$ python train_cifar.py –g -1 --dataset='cifar10'
```


## Single Node Performance Test Configurations

For Single Node Performance Test Configurations, please refer to following wiki:

https://github.com/intel/chainer/wiki/Intel-Chainer-Single-Node-Performance-Test-Configurations


## License

MIT License (see `LICENSE` file).


## Reference

Tokui, S., Oono, K., Hido, S. and Clayton, J.,
Chainer: a Next-Generation Open Source Framework for Deep Learning,
*Proceedings of Workshop on Machine Learning Systems(LearningSys) in
The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, (2015)
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)


## More Information
- [Intel® optimization for Chainer github](https://github.com/intel/chainer)
- [Release notes](https://github.com/intel/chainer/releases)
