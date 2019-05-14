import tensorflow as tf
import math

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.common import get_global_step_var
from tensorpack.models import (MaxPooling, Conv2D, GlobalAvgPooling, BatchNorm, FullyConnected, layer_register)
from dropblock import dropblock, dropblock2, dropblock3, dropblock4


def resnet_backbone(image, num_blocks, group_func, block_func, args):
    # if keep_probs is None:
    #     keep_probs = [None] * 4
    # if not isinstance(keep_probs, list) or len(keep_probs) != 4:
    #     raise ValueError('keep_probs is not valid:', keep_probs)
    keep_probs = [None] * 4
    dropblock_size = None

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
                total_steps = tf.cast(1281167//args.batch*args.lrs[-1], tf.float32)
                anchor_steps = tf.cast(1281167//args.batch*(args.lrs[1]+args.lrs[2])/2, tf.float32)
                current_step = tf.cast(get_global_step_var(), tf.float32)

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
        l = Conv2D('conv0', image, 64, 7, strides=2, activation=GNReLU)
        l = MaxPooling('pool0', l, pool_size=3, strides=2, padding='SAME')
        l = group_func('group0', l, block_func, 64 , num_blocks[0], 1, keep_prob=keep_probs[0], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2, keep_prob=keep_probs[1], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2, keep_prob=keep_probs[2], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2, keep_prob=keep_probs[3], dropblock_size=dropblock_size, groupsize=args.groupsize)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return logits


def resnet_group(name, l, block_func, features, count, stride, keep_prob, dropblock_size, groupsize):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, keep_prob=keep_prob, dropblock_size=dropblock_size, groupsize=groupsize)
    return l

def resnet_bottleneck(l, ch_out, stride, keep_prob, dropblock_size, groupsize, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    shortcut = dropblock4(shortcut, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize)
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=GNReLU)
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=GNReLU)
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_gn(zero_init=True))
    l = dropblock4(l, keep_prob=keep_prob, dropblock_size=dropblock_size, G=groupsize)
    out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_gn(zero_init=False))
    return tf.nn.relu(out)


@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    """
    https://arxiv.org/abs/1803.08494
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
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


def GNReLU(x, name=None):
    x = GroupNorm('gn', x)
    return tf.nn.relu(x, name=name)


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    n_in = l.get_shape().as_list()[1]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_gn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: GroupNorm('gn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: GroupNorm('gn', x)

