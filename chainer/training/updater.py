import copy
import multiprocessing

import six

from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module
from chainer import variable
from chainer import cuda
import chainer


class Updater(object):

    """Interface of updater objects for trainers.

    TODO(beam2d): document it.

    """

    def connect_trainer(self, trainer):
        """Connects the updater to the trainer that will call it.

        The typical usage of this method is to register additional links to the
        reporter of the trainer. This method is called at the end of the
        initialization of :class:`~chainer.training.Trainer`. The default
        implementation does nothing.

        Args:
            trainer (~chainer.training.Trainer): Trainer object to which the
                updater is registered.

        """
        pass

    def finalize(self):
        """Finalizes the updater object.

        This method is called at the end of training loops. It should finalize
        each dataset iterator used in this updater.

        """
        raise NotImplementedError

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Updater holds one or more optimizers with names. They can be retrieved
        by this method.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Optimizer of the name.

        """
        raise NotImplementedError

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        raise NotImplementedError

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        raise NotImplementedError

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        raise NotImplementedError


class StandardUpdater(Updater):

    """Standard implementation of Updater.

    This is the standard implementation of :class:`Updater`. It accepts one or
    more training datasets and one or more optimizers. The default update
    routine assumes that there is only one training dataset and one optimizer.
    Users can override this update routine by inheriting this class and
    overriding the :meth:`update_core` method. Each batch is converted to input
    arrays by :func:`~chainer.datasets.concat_examples` by default, which can
    also be manually set by ``converter`` argument.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary of iterators. If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            of optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`~chainer.dataset.concat_examples` is used
            by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        loss_func: Loss function. The target link of the main optimizer is used
            by default.

    Attributes:
        converter: Converter function.
        loss_func: Loss function. If it is ``None``, the target link of the
                   main optimizer is used instead.
        device: Device to which the training data is sent.
        iteration: Current number of completed updates.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 device=None, loss_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(optimizer, optimizer_module.Optimizer):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        self.loss_func = loss_func
        self.device = device
        self.iteration = 0

    @property
    def epoch(self):
        return self._iterators['main'].epoch

    @property
    def epoch_detail(self):
        return self._iterators['main'].epoch_detail

    @property
    def is_new_epoch(self):
        return self._iterators['main'].is_new_epoch

    def finalize(self):
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()

    def get_optimizer(self, name):
        return self._optimizers[name]

    def get_all_optimizers(self):
        return dict(self._optimizers)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~chainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        self.update_core()
        self.iteration += 1

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        if isinstance(in_arrays, tuple):
            in_vars = tuple(variable.Variable(x) for x in in_arrays)
            optimizer.update(loss_func, *in_vars)
        elif isinstance(in_arrays, dict):
            in_vars = {key: variable.Variable(x)
                       for key, x in six.iteritems(in_arrays)}
            optimizer.update(loss_func, **in_vars)
        else:
            in_var = variable.Variable(in_arrays)
            optimizer.update(loss_func, in_var)

    def serialize(self, serializer):
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)


class ParallelUpdater(StandardUpdater):

    """Implementation of a parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs.
    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.
        sel

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary of iterators. If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            of optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator is split equally between the
            devices and then passed with corresponding ``device`` option to
            this function. :func:`~chainer.dataset.concat_examples` is used by
            default.
        models: Dictionary of models. The main model should be the same model
            attached to the ``'main'`` optimizer.
        devices: Dictionary of devices to which the training data is sent. The
            devices should be arranged in a dictionary with the same structure
            as ``models``.
        loss_func: Loss function. The model is used as a loss function by
            default.

    """

    def __init__(self, iterator, optimizer, converter=convert.concat_examples,
                 models=None, devices=None, loss_func=None):
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            loss_func=loss_func,
        )

        if models is None:
            if devices is None:
                raise ValueError('either models or devices must be specified')
            names = list(six.iterkeys(devices))

            try:
                names.remove('main')
            except ValueError:
                raise KeyError("'devices' must contain a 'main' key.")

            models = {'main': optimizer.target}
            for name in names:
                model = copy.deepcopy(optimizer.target)
                if devices[name] >= 0:
                    model.to_gpu(devices[name])
                models[name] = model
            if devices['main'] >= 0:
                optimizer.target.to_gpu(devices['main'])

        self._devices = devices
        self._models = models

    def connect_trainer(self, trainer):
        # Add observers for all (other) models.
        model_main = self.get_optimizer('main').target
        models_others = {
            k: v for k, v in self._models.items() if v != model_main
        }
        for name, model in models_others.items():
            trainer.reporter.add_observer(name, model)

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        batch = self.get_iterator('main').next()

        #
        # Split the batch to sub-batches.
        #
        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        # For reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        for model_key, model in six.iteritems(self._models):
            in_arrays = in_arrays_list[model_key]
            loss_func = self.loss_func or model
            losses.append(loss_func, in_arrays)

        # For _uninitialized_params
        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward()

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)


def _calc_loss(model, in_arrays):
    if isinstance(in_arrays, tuple):
        in_vars = tuple(variable.Variable(x) for x in in_arrays)
        return model(*in_vars)
    elif isinstance(in_arrays, dict):
        in_vars = {key: variable.Variable(x)
                   for key, x in six.iteritems(in_arrays)}
        return model(**in_vars)
    else:
        in_vars = variable.Variable(in_arrays)
        return model(in_vars)


use_gather = True


def _sync_grad(comm, model):
    global streams
    from cupy.cuda import nccl
    null_stream = cuda.Stream.null
    if use_gather:
        gg = model.gather_grads()
        # NCCL: allreduce
        comm.allReduce(gg.data.ptr, gg.data.ptr, gg.size,
                       nccl.NCCL_FLOAT, nccl.NCCL_SUM, null_stream.ptr)
        model.scatter_grads(gg)
    else:
        for param in enumerate(model.params()):
            grad = param.grad
            assert isinstance(grad, cuda.ndarray)
            assert grad.flags.c_contiguous
            comm.allReduce(grad.data.ptr, grad.data.ptr, grad.nbytes,
                           nccl.NCCL_FLOAT, nccl.NCCL_SUM, null_stream.ptr)


class _Worker(multiprocessing.Process):
    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = master._master
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.optimizer = master.get_optimizer('main')
        self.n_devices = len(master._devices)

    def setup(self):
        from cupy.cuda import nccl
        print("setup %d" % self.device)
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = chainer.reporter.Reporter()
        self.reporter.add_observer('main', self.model)
        print("%d setup done" % self.device)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'update':
                # For reducing memory
                self.model.cleargrads()

                batch = self.converter(self.iterator.next(), self.device)
                observation = {}
                with self.reporter.scope(observation):
                    loss = _calc_loss(self.model, batch)

                self.model.cleargrads()
                loss.backward()

                _sync_grad(self.comm, self.model)
                self.optimizer.update()

                # self.pipe.send(observation)


class MultiprocessParallelUpdater(StandardUpdater):
    """Implementation of a multiprocess parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs
    with multi-process data parallelism. The GPUs provided are placed in
    a tree structure of masters and workers, each with its own process.

    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    Unlike other built-in Updater classes, the model (attached to the
    optimizer) must be the loss function.

    Args:
        iterators: Lsit of dataset iterator for the training dataset.
        optimizer: Optimizer to update parameters. The model should be attached
            to the optimizer.
        converter: Converter function to build input arrays. Each batch
            extracted by the iterator is split equally between the devices and
            then passed with corresponding ``device`` option to this function.
            :func:`~chainer.dataset.concat_examples` is used by default.
        devices: Dictionary or list of devices to which the training data is
            sent. The master device will be the first one in the list or the
            value attached to the key ``'main'``.

    """

    def __init__(self, iterators, optimizer, converter=convert.concat_examples,
                 devices=None):

        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps /= len(devices)
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps /= len(devices) ** 2
        else:
            optimizer.lr /= len(devices)

        super(MultiprocessParallelUpdater, self).__init__(
            iterator=iterators[0],
            optimizer=optimizer,
            converter=converter
        )

        if isinstance(devices, dict):
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        if devices is None or any(device is None for device in devices):
            raise ValueError('must specify GPU devices')

        self._master = optimizer.target
        self._devices = devices
        self._mpu_iterators = iterators
        self._initialized = False

        self._pipes = []
        self._workers = []
        self.comm = None

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        print("setup_workers()")

        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self._devices[0]):
            from cupy.cuda import nccl
            self._master.to_gpu(self._devices[0])
            comm_id = nccl.get_unique_id()
            self._send_message(("set comm_di", comm_id))
            self.comm = nccl.NcclCommunicator(len(self._devices), comm_id, 0)

        print("setup_workers() done")

    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            batch = self.converter(batch, self._devices[0])

            loss = _calc_loss(self._master, batch)

            self._master.cleargrads()
            loss.backward()

            _sync_grad(self.comm, self._master)

            optimizer.update()

            # for pipe in self._pipes:
            #     chainer.reporter.report(pipe.recv())

    def finalize(self):
        self._send_message(('finalize', None))

        for worker in self._workers:
            print("join", worker)
            worker.join()

