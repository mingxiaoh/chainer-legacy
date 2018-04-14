<div align="center"><img src="docs/image/chainer_red_h.png" width="400"/></div>

# IntelChainer: Optimized-Chainer for Intel Architectures

[![GitHub license](https://img.shields.io/github/license/intel/chainer.svg)](https://github.com/intel/chainer)
[![travis](https://img.shields.io/travis/intel/chainer/master.svg)](https://travis-ci.org/intel/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)


Chainer* is a Python*-based deep learning framework aiming at flexibility and intuition. It provides automatic differentiation APIs based on the define-by-run approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. It supports various network architectures including feed-forward nets, convnets, recurrent nets and recursive nets. It also supports per-batch architectures. Forward computation can include any control flow statements of Python without lacking the ability of backpropagation. It makes code intuitive and easy to debug. Intel® optimization for Chainer, is currently integrated with the latest release of Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) 2017 optimized for Intel® Advanced Vector Extensions 2 (Intel® AVX) and Intel® Advanced Vector Extensions 512 (Intel®AVX-512) instructions which are supported in Intel® Xeon® and Intel® Xeon Phi™ processors.

## Recommended Environments
We recommend these Linux distributions.
- Ubuntu 16.04 LTS 64bit
- CentOS 7 64bit

The following versions of Python can be used: 
- 2.7.10+, 3.5.2+, and 3.6.0+

Above recommended environments are tested. We cannot guarantee that Intel® optimization for Chainer works on other environments including Windows* and macOS*, even if Intel optimization for Chainer looks to be running correctly.


## Install Chainer from source
You can use setup.py to install Chainer from the tarball:

```sh
$ python setup.py install
```
ideep4py has been splitted from Chainer, so you also need to install ideep4py:
```sh
$ pip install ideep4y
```
Use pip to uninstall chainer and ideep4py:

```sh
$ pip uninstall chainer ideep4py
```

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
