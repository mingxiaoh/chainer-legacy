<div align="center"><img src="docs/image/chainer_red_h.png" width="400"/></div>

# IntelChainer: Optimized-Chainer for Intel Architectures

[![GitHub license](https://img.shields.io/github/license/chainer/chainer.svg)](https://github.com/intel/chainer)
[![travis](https://img.shields.io/travis/intel/chainer/master_v3.svg)](https://travis-ci.org/intel/chainer)
[![Read the Docs](https://readthedocs.org/projects/chainer/badge/?version=stable)](https://docs.chainer.org/en/stable/?badge=stable)


*Chainer* is a Python-based deep learning framework aiming at flexibility.
It provides automatic differentiation APIs based on the **define-by-run** approach (a.k.a. dynamic computational graphs) as well as object-oriented high-level APIs to build and train neural networks. IntelChainer is optimized-chainer for Intel architectures.

## Recommended Environments
We recommend these Linux distributions.
- Ubuntu 14.04/16.04 LTS 64bit
- CentOS 7 64bit

The following versions of Python can be used: 
- 2.7.5+, 3.5.2+, and 3.6.0+


## Dependencies
Before installing Chainer, we recommend to upgrade setuptools if you are using an old one:

$ pip install -U setuptools

The following packages are required to install Chainer.
- NumPy 1.9, 1.10, 1.11, 1.12, 1.13
- Six 1.9+
- Swig 3.0.9+

The following packages are optional dependencies. Chainer can be installed without them, in which case the corresponding features are not available.

CUDA/cuDNN support
- cupy 2.0+

Caffe model support
- protobuf 3.0+

Image dataset support
- pillow 2.3+

HDF5 serialization support
- h5py 2.5+


## Install Chainer from source
You can use setup.py to install Chainer from the tarball:

```sh
$ git clone https://github.com/chainer/chainer.git
$ cd chainer
$ python setup.py install
```

Use pip to uninstall Chainer:

```sh
$ pip uninstall chainer
```

## Install CUDA
To enable CUDA support, [set up CUDA](http://docs.nvidia.com/cuda/index.html#installation-guides) and install [CuPy](https://github.com/cupy/cupy).

```sh
$ pip install cupy
```


## Run with Docker

We provide the Dockerfile for cpu in chainer/docker directory based on python2 and python3, respectively. You can refer to wiki

https://github.com/intel/chainer/wiki/How-to-build-and-run-Intel-Chainer-Docker-image

to check how to build/run with docker.


## License

MIT License (see `LICENSE` file).


## Reference

Tokui, S., Oono, K., Hido, S. and Clayton, J.,
Chainer: a Next-Generation Open Source Framework for Deep Learning,
*Proceedings of Workshop on Machine Learning Systems(LearningSys) in
The Twenty-ninth Annual Conference on Neural Information Processing Systems (NIPS)*, (2015)
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)


## More Information

- Official site: http://chainer.org/
- Official document: http://docs.chainer.org/
- Pfn chainer github: https://github.com/pfnet/chainer
- Intel chainer github: https://github.com/intel/chainer
- Forum: https://groups.google.com/forum/#!forum/chainer
- Forum (Japanese): https://groups.google.com/forum/#!forum/chainer-jp
- Twitter: https://twitter.com/ChainerOfficial
- Twitter (Japanese): https://twitter.com/chainerjp
- External examples: https://github.com/pfnet/chainer/wiki/External-examples
- Research projects using Chainer: https://github.com/pfnet/chainer/wiki/Research-projects-using-Chainer
