import argparse
import numpy as np
import os
import sys
import time
from chainer import optimizers
from models import get_model

archs = ["alexnet", "googlenet", "vgga", "resnet50", "vgg16", "overfeat"]


def __log__(logger, msg):
    logger.write(msg)
    logger.flush()


def test_performance(modelName, data_size, batch_size, insize, epoch, log):
    __log__(log, "Performance test for {0}...\n".format(modelName))
    # Assign the insize for each model
    model = get_model(modelName)
    model.to_ia()
    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)

    # Init the timer
    total_forward = 0
    total_backward = 0

    n_dry = 3
    count = 0

    # Start to iterate
    for i in range(epoch):
        __log__(log, "Iteration: {0}...\n".format(i))
        __log__(log, "Init the test data...\n")
        data = np.random.uniform(
            -1, 1, (batch_size, 3, insize, insize)).astype(np.float32)
        x = np.asarray(data)

        __log__(log, "\tForward...\n")

        if modelName == "googlenet":
            start_time = time.time()
            out1, out2, out3 = model.forward(x)
            end_time = time.time()
            out = out1 + out2 + out3
        else:
            start_time = time.time()
            out = model.forward(x)
            end_time = time.time()

        time_ = (end_time - start_time) * 1000
        if i > n_dry - 1:
            total_forward += time_
            count += 1

        __log__(log, "\tForward complete...\n")

        out.zerograd()
        out.grad = np.random.uniform(
            -1, 1, out.grad.shape).astype(out.grad.dtype)
        model.cleargrads()

        __log__(log, "\tBackward...\n")

        start_time = time.time()
        out.backward()
        end_time = time.time()
        time_ = (end_time - start_time) * 1000
        if i > n_dry - 1:
            total_backward += time_

        __log__(log, "\tBackward complete...\n")

        if modelName == "googlenet":
            del out1, out2, out3

        del out

        __log__(log, "\tIteration {0} complete...\n".format(i))

    del model

    __log__(log, "Average forward (ms): {0}\n".format(total_forward / count))
    __log__(log, "Average backward (ms): {0}\n".format(total_backward / count))
    __log__(log, "Average total (ms): {0}\n".format(
            (total_forward + total_backward) / count))

    result = {
        "arch_name": modelName,
        "config": "Opt.",
        "batch_size": batch_size,
        "shape": "3*{0}*{1}".format(insize, insize),
        "iterations": data_size / batch_size,
        "epoch": epoch,
        "Forward": round(data_size * 1000 * count / total_forward, 3),
        "Backward": round(data_size * 1000 * count / total_backward, 3),
        "Total": round(data_size * 1000 * count /
                       (total_forward + total_backward), 3)
    }

    __log__(log, "Performance test for {0} complete...\n".format(modelName))
    return result


if __name__ == "__main__":
    # parse the opts and args
    parser = argparse.ArgumentParser(description="""
    This script is for the performance test.
    You can use it to perform test by specifying -a, -b and -i.
    """)
    parser.add_argument("--arch", "-a",
                        help="Architectures: alex, googlenet, vgga, overfeat",
                        required=True)
    parser.add_argument("--batchsize", "-b", type=int,
                        help="Minibatch size", required=True)
    parser.add_argument("--insize", "-i", type=int,
                        help="Insize: Set the w*h = insize*insize",
                        required=True)
    parser.add_argument("--datasize", "-d", type=int,
                        help="Dataset size", required=True)
    parser.add_argument("--epoch", "-e", type=int, help="Epoch", required=True)
    parser.add_argument("--log", "-l", default=None,
                        help="Log file path. If empty, will use stderr.")
    args = parser.parse_args()

    if args.log:
        args.log = os.path.abspath(args.log)

    if args.arch in archs:
        log = sys.stderr
        if args.log:
            log = open(args.log, 'w')

        __log__(log, "arch: {0}, batchsize: {1}, datasize: {2},\
                epoch: {3}, insize: {4}\n".format(args.arch,
                                                  args.batchsize,
                                                  args.datasize,
                                                  args.epoch,
                                                  args.insize))
        test_result = test_performance(args.arch, args.datasize,
                                       args.batchsize, args.insize,
                                       args.epoch, log)
        __log__(log, "performance test result: {0}\n".format(test_result))
        if args.log:
            log.close()
    else:
        raise ValueError("Arch ERROR!")
