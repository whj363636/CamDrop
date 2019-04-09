# -*- coding: utf-8 -*-
# File: dropblock.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
# from tensorpack.tfutils.compat import tfv1 as tf  # this should be avoided first in model code
from tensorpack.tfutils.tower import get_current_tower_context
import tensorflow as tf

__all__ = ['dropblock', 'dropblock2','dropblock3', 'dropblock11'] # 1: paper baseline; 2: group dropout; 3: shuffle group dropout  

def dropblock(net, keep_prob, dropblock_size, data_format='channels_first'):
  """DropBlock: a regularization method for convolutional neural networks.
  DropBlock is a form of structured dropout, where units in a contiguous
  region of a feature map are dropped together. DropBlock works better than
  dropout on convolutional layers due to the fact that activation units in
  convolutional layers are spatially correlated.
  See https://arxiv.org/pdf/1810.12890.pdf for details.
  Args:
    net: `Tensor` input tensor.
    is_training: `bool` for whether the model is training.
    keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock. "None"
        means no DropBlock.
    dropblock_size: `int` size of blocks to be dropped by DropBlock.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
      A version of input tensor with DropBlock applied.
  Raises:
      if width and height of the input tensor are not equal.
  """

  ctx = get_current_tower_context()
  is_training = bool(ctx.is_training)

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, _ = net.get_shape().as_list()
  else:
    _, _, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (
      width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(
      valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(tf.shape(net), dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(
        block_pattern,
        axis=[1, 2] if data_format == 'channels_last' else [2, 3],
        keepdims=True)
  else:
    if data_format == 'channels_last':
      ksize = [1, dropblock_size, dropblock_size, 1]
    else:
      ksize = [1, 1, dropblock_size, dropblock_size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME',
        data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(
      tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
      block_pattern, net.dtype)
  return net

def dropblock2(net, keep_prob, dropblock_size, G=32, data_format='channels_first'):
  """
  mimic GN
  """

  ctx = get_current_tower_context()
  is_training = bool(ctx.is_training)

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(dropblock_size, net.shape))

  if data_format == 'channels_last':
    N, height, width, C = net.get_shape().as_list()
  else:
    N, C, height, width = net.get_shape().as_list()
  N = tf.shape(net)[0] 
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  net = tf.reshape(net, [N, G, C//G, height, width]) 

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0) # for depth
  valid_block_center = tf.expand_dims(valid_block_center, 0) # for batch
  valid_block_center = tf.expand_dims(valid_block_center, 0) # for channel
  randnoise = tf.random_uniform([N, G, 1, width, height], dtype=tf.float32)

  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(block_pattern, axis=[2, 3, 4], keepdims=True)
  else:
    ksize = [1, 1, dropblock_size, dropblock_size]
    
    block_pattern = tf.reduce_max(-block_pattern, reduction_indices=[2])
    block_pattern = -tf.nn.max_pool(block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    block_pattern = tf.expand_dims(block_pattern, 2)

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)
  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  net = tf.reshape(net, [N, height, width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height, width])
  return net


def dropblock3(net, keep_prob, dropblock_size, G=32, data_format='channels_first'):
  """
  shuffle group dropout
  """

  ctx = get_current_tower_context()
  is_training = bool(ctx.is_training)

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(dropblock_size, net.shape))

  if data_format == 'channels_last':
    N, height, width, C = net.get_shape().as_list()
  else:
    N, C, height, width = net.get_shape().as_list()
  N = tf.shape(net)[0] 
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')

  net = tf.transpose(net, [1, 0, 2, 3]) # [C, N, height, width]
  net = tf.random_shuffle(net)
  net = tf.transpose(net, [1, 0, 2, 3]) # [N, C, height, width]
  net = tf.reshape(net, [N, G, C//G, height, width]) 

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (width - dropblock_size + 1)**2

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0) # for depth
  valid_block_center = tf.expand_dims(valid_block_center, 0) # for batch
  valid_block_center = tf.expand_dims(valid_block_center, 0) # for channel
  randnoise = tf.random_uniform([N, G, 1, width, height], dtype=tf.float32)

  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(block_pattern, axis=[2, 3, 4], keepdims=True)
  else:
    ksize = [1, 1, dropblock_size, dropblock_size]
    
    block_pattern = tf.reduce_max(-block_pattern, reduction_indices=[2])
    block_pattern = -tf.nn.max_pool(block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    block_pattern = tf.expand_dims(block_pattern, 2)

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)
  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  net = tf.reshape(net, [N, height, width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height, width])
  return net




def dropblock11(net, keep_prob, dropblock_size=7, data_format='channels_first'):
  """ 
  slice channel and H,W random select in each max-pooling
  """

  ctx = get_current_tower_context()
  is_training = bool(ctx.is_training)

  if not is_training or keep_prob is None:
    return net

  if data_format == 'channels_last':
    _, width, height, chann = net.get_shape().as_list()
  else:
    _, chann, width, height = net.get_shape().as_list()

  chann = min(2, chann)
  slices = tf.split(net, num_or_size_splits=chann, axis=-1 if data_format == 'channels_last' else 1)

  for i in range(slices):
    dropblock_w = max(1, width)
    dropblock_h = max(1, height)
    # seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (1.0 - keep_prob) * width*height / (dropblock_w*dropblock_h)  / ((width - dropblock_w + 1)*(height - dropblock_h + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
    valid_block_center = tf.logical_and(tf.logical_and(w_i >= int(dropblock_w // 2), w_i < width - (dropblock_w - 1) // 2), 
                                        tf.logical_and(h_i >= int(dropblock_h // 2), h_i < width - (dropblock_h - 1) // 2))
    valid_block_center = tf.expand_dims(valid_block_center, 0)
    valid_block_center = tf.expand_dims(valid_block_center, -1 if data_format == 'channels_last' else 0)

    # generate noise and block_pattern
    randnoise = tf.random_uniform(tf.shape(slices[i]), dtype=tf.float32)
    block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
        (1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if dropblock_w == width and dropblock_h == height:
      block_pattern = tf.reduce_min(block_pattern, axis=[1, 2] if data_format == 'channels_last' else [2, 3], keepdims=True)
    else:
      if data_format == 'channels_last':
        ksize = [1, dropblock_w, dropblock_h, 1]
      else:
        ksize = [1, 1, dropblock_w, dropblock_h]
      block_pattern = -tf.nn.max_pool(-block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC' if data_format == 'channels_last' else 'NCHW')
    slices[i] = block_pattern

  percent_ones = tf.cast(tf.reduce_sum((tf.concat(slices, axis=-1 if data_format == 'channels_last' else 1))), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  return net