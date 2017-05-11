# Chainer: a neural network framework

## Requirements and Installation

Please reference README.md in branch master.

## Static Graph

### Reference

Static graph prototype of Chainer

```
https://github.com/bkvogel/static-define-by-run
```

### Branch Description

Current branch implements a preliminary integration of static graph prototype. The integration illustrates how to apply static graph feature to Linear function optimized by Intel MKL-DNN (Intel Math Kernel Library for Deep Neural Networks).

### Usage

```
cd tests/chainer_tests/ && python test_static_graph.py
```

## License

MIT License (see `LICENSE` file).
