#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-convnet.py

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.models import (MaxPooling, Conv2D, GlobalAvgPooling, BatchNorm, FullyConnected, layer_register)
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.common import get_global_step_var

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import summary
from dropblock import dropblock, dropblock2, CamDrop

"""
MNIST ConvNet example.
about 0.6% validation error after 30 epochs.
"""

IMAGE_SIZE = 28
NUM_UNITS = None
BATCH_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('-n', '--num_units', help='number of units in each stage', type=int, default=1)
parser.add_argument('--load', help='load model for training')
parser.add_argument('--dropblock_groups', type=str, default='3', help='which group to drop')
parser.add_argument('--blocksize', type=int, default=7, help='The size of dropblock.')
parser.add_argument('--groupsize', type=int, default=16, help='The size of dropgroup.')
parser.add_argument('--norm', type=str, default='BN', help='BN or GN')
parser.add_argument('--strategy', type=str, default='decay', help='strategy for dropblock, decay or not')

parser.add_argument('--keep_prob', type=float, default=None, help='The keep probabiltiy of dropblock.')
parser.add_argument('--arch', type=str, default=None, help='L or R')
parser.add_argument('--lr', type=float, default=None, help='learning rate')
parser.add_argument('--ablation', type=str, default='', help='.')

args = parser.parse_args()

class Model(ModelDesc):
    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n
        self.norm = BNReLU if args.norm == 'BN' else tf.nn.relu
        self.flag = -1

    def inputs(self):
        """
        Define all the inputs (with type, shape, name) that the graph will need.
        """
        return [tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE), tf.float32, 'input'),
                tf.TensorSpec((None,), tf.int32, 'label')]

    def build_graph(self, image, label):
        """This function should build the model which takes the input variables
        and return cost at the end"""

        # In tensorflow, inputs to convolution function are assumed to be
        # NHWC. Add a single channel here.
        image = tf.expand_dims(image, 3)
        image = tf.transpose(image, [0, 3, 1, 2])

        image = image * 2 - 1   # center the pixels values at zero
        # The context manager `argscope` sets the default option for all the layers under
        # this context. Here we use 32 channel convolution with shape 3x3
        keep_probs = [None] * 3
        dropblock_size = None
        self.flag += 1

        if args.dropblock_groups:
            dropblock_size = args.blocksize
            # Computes DropBlock keep_probs for different block groups of ResNet.
            dropblock_groups = [int(x) for x in args.dropblock_groups.split(',')]
            for block_group in dropblock_groups:
                if block_group < 1 or block_group > 3:
                    raise ValueError('dropblock_groups should be a comma separated list of integers ' 'between 1 and 3 (dropblock_groups: {}).' .format(args.dropblock_groups))
                if args.strategy == None or args.strategy == '1':
                    keep_probs[block_group - 1] = args.keep_prob
                else:
                    total_steps = tf.cast(60000//BATCH_SIZE*args.lr, tf.float32)
                    current_step = tf.cast(get_global_step_var(), tf.float32)

                    if args.strategy == 'decay':
                        # Scheduled keep_probs for DropBlock.
                        current_ratio = current_step / total_steps
                        keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                        keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(3 - block_group))  
                    elif args.strategy == 'V':
                        # V-scheduled keep_probs for DropBlock.
                        # Scheduled keep_probs for DropBlock.
                        current_ratio = current_step / total_steps
                        keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                        keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(3 - block_group))  

        if args.arch == 'L':
            with argscope([Conv2D, GlobalAvgPooling], kernel_size=3, activation=tf.nn.relu, filters=32):
                l = LinearWrap(image)
                l = Conv2D('conv0', l)
                l = MaxPooling('pool0', l, 2)
                l = dropblock4(l, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                l = Conv2D('conv1', l)
                l = Conv2D('conv2', l)
                l = MaxPooling('pool1', l, 2)
                l = dropblock4(l, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                l = Conv2D('conv3', l)
                l = GlobalAvgPooling('gap', l)
                logits = FullyConnected('linear', l, 10)


        elif args.arch == 'R':
            def residual(name, l, keep_prob, increase_dim=False, first=False):
                shape = l.get_shape().as_list()
                in_channel = shape[1]

                if increase_dim:
                    out_channel = in_channel * 2
                    stride1 = 2
                else:
                    out_channel = in_channel
                    stride1 = 1

                with tf.variable_scope(name):
                    shortcut = l if first or increase_dim else dropblock4(l, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                    b1 = l if first else self.norm(l)
                    c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=self.norm)
                    c1 = dropblock4(c1, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                    c2 = Conv2D('conv2', c1, out_channel)
                    c2 = dropblock4(c2, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)

                    if increase_dim:
                        shortcut = AvgPooling('pool', shortcut, 2)
                        shortcut = dropblock4(shortcut, label=label, flag=self.flag, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                        shortcut = tf.pad(shortcut, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                    l = c2 + shortcut
                    return l

            with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), argscope(Conv2D, use_bias=False, kernel_size=3, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
                l = Conv2D('conv0', image, 16, activation=self.norm)
                l = residual('res1.0', l, keep_probs[0], first=True)
                l = residual('res1.{}'.format(1), l, keep_probs[0])
                # 32,c=16

                l = residual('res2.0', l, keep_probs[1], increase_dim=True)
                l = residual('res2.{}'.format(1), l, keep_probs[1])
                # 16,c=32

                l = residual('res3.0', l, keep_probs[2], increase_dim=True)
                l = residual('res3.' + str(1), l, keep_probs[2])
                l = self.norm('bnlast', l)
                # 8,c=64
                l = GlobalAvgPooling('gap', l)

            logits = FullyConnected('linear', l, 10)

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32, name='correct')
        accuracy = tf.reduce_mean(correct, name='accuracy')

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        train_error = tf.reduce_mean(1 - correct, name='train_error')
        summary.add_moving_summary(train_error, accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        wd_cost = tf.multiply(1e-5,
                              regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        total_cost = tf.add_n([wd_cost, cost], name='total_cost')
        summary.add_moving_summary(cost, wd_cost, total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=args.lr, # 1e-3
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr)


def get_data():
    train = BatchData(dataset.Mnist('train'), 128)
    test = BatchData(dataset.Mnist('test'), 256, remainder=True)

    train = PrintData(train)

    return train, test


if __name__ == '__main__':
    # automatically setup the directory train_log/mnist-convnet for logging
    logger.set_logger_dir(os.path.join('train_log', 'MNIST-arch{}-lr{}-drop{}-method{}'.format(args.arch, args.lr, args.keep_prob, args.ablation)))

    dataset_train, dataset_test = get_data()

    # How many iterations you want in each epoch.
    # This len(data) is the default value.
    steps_per_epoch = len(dataset_train)

    # get the config which contains everything necessary in a training
    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=FeedInput(dataset_train),
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                ScalarStats(    # produce `val_accuracy` and `val_cross_entropy_loss`
                    ['cross_entropy_loss', 'accuracy'], prefix='val')),
            # MaxSaver has to come after InferenceRunner
            MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )
    launch_train_with_config(config, SimpleTrainer())
