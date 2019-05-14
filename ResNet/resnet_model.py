# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf

from tensorpack.models import BatchNorm, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.common import get_global_step_var
from dropblock import dropblock, dropblock2, dropblock3, dropblock4

def resnet_backbone(image, label, num_blocks, group_func, block_func, flag, args):
    # if keep_probs is None:
    #     keep_probs = [None] * 4
    # if not isinstance(keep_probs, list) or len(keep_probs) != 4:
    #     raise ValueError('keep_probs is not valid:', keep_probs)
    keep_probs = [None] * 4
    dropblock_size = None
    flag +=1

    if args.dropblock_groups:
        dropblock_size = args.blocksize
        # Computes DropBlock keep_probs for different block groups of ResNet.
        dropblock_groups = [int(x) for x in args.dropblock_groups.split(',')]
        for block_group in dropblock_groups:
            if block_group < 1 or block_group > 4:
                raise ValueError('dropblock_groups should be a comma separated list of integers ' 'between 1 and 4 (dropblock_groups: {}).' .format(args.dropblock_groups))
            if args.strategy == None or args.strategy == '':
                keep_probs[block_group - 1] = args.keep_prob
            else:
                current_step = tf.cast(get_global_step_var(), tf.float32)
                total_steps = tf.cast(1281167//args.batch*args.lrs[-1], tf.float32)
                anchor_steps = tf.cast(1281167//args.batch*(args.lrs[1]+args.lrs[2])/2, tf.float32)

                if args.strategy == 'decay':
                    # Scheduled keep_probs for DropBlock.
                    current_ratio = current_step / total_steps
                    keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                    keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(4 - block_group))  
                elif args.strategy == 'V':
                    # V-scheduled keep_probs for DropBlock.
                    current_ratio = tf.cond(tf.less(current_step, anchor_steps), lambda: current_step / anchor_steps, lambda: (total_steps - current_step) / (total_steps - anchor_steps))
                    keep_prob = (1 - current_ratio * (1 - args.keep_prob))
                    keep_probs[block_group - 1] = 1 - ((1 - keep_prob) / 4.0**(4 - block_group))  

    with argscope(Conv2D, use_bias=False, kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that this pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=BNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, label, flag, block_func, 64 , num_blocks[0], 1, keep_prob=keep_probs[0], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group1', l, label, flag, block_func, 128, num_blocks[1], 2, keep_prob=keep_probs[1], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group2', l, label, flag, block_func, 256, num_blocks[2], 2, keep_prob=keep_probs[2], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group3', l, label, flag, block_func, 512, num_blocks[3], 2, keep_prob=keep_probs[3], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000, kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    return logits


def resnet_group(name, l, label, flag, block_func, features, count, stride, keep_prob, dropblock_size, groupsize):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, label, flag, features, stride if i == 0 else 1, keep_prob=keep_prob, dropblock_size=dropblock_size, groupsize=groupsize)
    return l

def resnet_bottleneck(l, label, flag, ch_out, stride, keep_prob, dropblock_size, groupsize, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    shortcut = dropblock4(shortcut, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize, label=label, flag=flag)
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize, label=label, flag=flag)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize, label=label, flag=flag)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize, label=label, flag=flag)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)

def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)