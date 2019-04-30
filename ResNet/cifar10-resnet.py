#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu

import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.models import (MaxPooling, Conv2D, GlobalAvgPooling, BatchNorm, FullyConnected, layer_register)
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.common import get_global_step_var
from dropblock import dropblock, dropblock2, dropblock3, dropblock4

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.

To train:
    ./cifar10-resnet.py --gpu 0,1
"""


BATCH_SIZE = 128
NUM_UNITS = None

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('-n', '--num_units', help='number of units in each stage', type=int, default=18)
parser.add_argument('--load', help='load model for training')
parser.add_argument('--lrs', nargs='+', default=[82, 123, 300, 400], type=int, help='')
parser.add_argument('--dropblock_groups', type=str, default='1,2,3', help='which group to drop')

parser.add_argument('--blocksize', type=int, default=8, help='The size of dropblock.')
parser.add_argument('--keep_prob', type=float, default=None, help='The keep probabiltiy of dropblock.')

parser.add_argument('--start', type=int, default=1, help='The start epoch.')
parser.add_argument('--groupsize', type=int, help='The size of dropgroup.')
parser.add_argument('--norm', type=str, default='BN', help='BN or GN')
parser.add_argument('--strategy', type=str, default=None, help='strategy for dropblock, decay or not')
parser.add_argument('--ablation', type=str, default='', help='.')

args = parser.parse_args()


def GroupNorm(x, group, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')


@layer_register(use_scope=None)
def GNReLU(x, name=None):
    x = GroupNorm(x, 8)
    return tf.nn.relu(x, name=name)

class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n
        self.norm = BNReLU if args.norm == 'BN' else GNReLU

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        keep_probs = [None] * 3
        dropblock_size = None

        if args.dropblock_groups:
            dropblock_size = args.blocksize
            # Computes DropBlock keep_probs for different block groups of ResNet.
            dropblock_groups = [int(x) for x in args.dropblock_groups.split(',')]
            for block_group in dropblock_groups:
                if block_group < 1 or block_group > 3:
                    raise ValueError('dropblock_groups should be a comma separated list of integers ' 'between 1 and 3 (dropblock_groups: {}).' .format(args.dropblock_groups))
                if args.strategy == None or args.strategy == '':
                    keep_probs[block_group - 1] = args.keep_prob
                else:
                    total_steps = tf.cast(50000//BATCH_SIZE*args.lrs[-1], tf.float32)
                    anchor_steps = tf.cast(50000//BATCH_SIZE*(args.lrs[1]+args.lrs[2])/2, tf.float32)
                    current_step = tf.cast(get_global_step_var(), tf.float32)

                    if args.strategy == 'decay':
                        # Scheduled keep_probs for DropBlock.
                        current_ratio = current_step / total_steps
                        keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                        keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(3 - block_group))  
                    elif args.strategy == 'V':
                        # V-scheduled keep_probs for DropBlock.
                        current_ratio = tf.cond(tf.less(current_step, anchor_steps), lambda: current_step / anchor_steps, lambda: (total_steps - current_step) / (total_steps - anchor_steps))
                        keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                        keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(3 - block_group))  

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
                shortcut = l if first or increase_dim else dropblock3(l, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                b1 = l if first else self.norm(l)
                c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=self.norm)
                c1 = dropblock3(c1, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                c2 = Conv2D('conv2', c1, out_channel)
                c2 = dropblock3(c2, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)

                if increase_dim:
                    shortcut = AvgPooling('pool', shortcut, 2)
                    shortcut = dropblock3(shortcut, keep_prob=keep_prob, dropblock_size=args.blocksize, G=args.groupsize)
                    shortcut = tf.pad(shortcut, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + shortcut
                return l

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), argscope(Conv2D, use_bias=False, kernel_size=3, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, activation=self.norm)
            l = residual('res1.0', l, keep_probs[0], first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l, keep_probs[0])
            # 32,c=16

            l = residual('res2.0', l, keep_probs[1], increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l, keep_probs[1])
            # 16,c=32

            l = residual('res3.0', l, keep_probs[2], increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l, keep_probs[2])
            l = self.norm('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, 10)
        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean(('train',))
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir(
        os.path.join('train_log',
                     'cifar10-norm{}-drop{}-groupsize{}-{}'.format(
                        args.norm, args.keep_prob, args.groupsize, args.ablation)))

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (args.lrs[0], 0.01), (args.lrs[1], 0.001), (args.lrs[2], 0.0002)])
        ],
        starting_epoch=args.start,
        max_epoch=args.lrs[3],
        session_init=SaverRestore(args.load) if args.load else None
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
