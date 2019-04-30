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

__all__ = ['dropblock', 'dropblock2','dropblock3','dropblock4'] # 1: paper baseline; 2: group dropout; 3: group soft-dropout; 4: Uout group dropout  

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


def dropblock2(net, keep_prob, dropblock_size, G, data_format='channels_first'):
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
  # seed_drop_rate = (1.0 - keep_prob) * width**2 * G**2 / (C * dropblock_size**2) / (C * (width - dropblock_size + 1)**2)
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


def dropblock3(net, keep_prob, dropblock_size, G, data_format='channels_first'):
  """
  adaptive GN (softmax)
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

  # net_max = tf.nn.avg_pool(net, ksize=[1, 1, dropblock_size, dropblock_size], strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
  net_max = tf.nn.avg_pool(net, ksize=[1, 1, dropblock_size, dropblock_size], strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
  if data_format == 'channels_last':
    _, new_height, new_width, _ = net_max.get_shape().as_list()
  else:
    _, _, new_height, new_width = net_max.get_shape().as_list()

  net_max = tf.reshape(net_max, [N, G, C//G, new_height, new_width]) 
  # net_max = tf.reduce_max(net_max, reduction_indices=[2])
  net_max = tf.reduce_mean(net_max, reduction_indices=[2])
  mask_max = tf.reshape(tf.nn.softmax(tf.reshape(net_max, [N, G, -1])), [N, G, new_height, new_width])

  left_or_top = (dropblock_size-1) // 2 
  right_or_bot = left_or_top if dropblock_size % 2 == 1 else dropblock_size-left_or_top-1
  mask_max = tf.pad(mask_max, [[0, 0], [0, 0], [left_or_top, right_or_bot], [left_or_top, right_or_bot]])
  mask_max = tf.expand_dims(mask_max, 2)

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
  randnoise = tf.random_uniform([N, G, 1, height, width], dtype=tf.float32)
  # fadeoff = tf.random_uniform([N, G, 1, width, height], minval=-seed_drop_rate, maxval=seed_drop_rate, dtype=tf.float32)


  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise + mask_max) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(block_pattern, axis=[2, 3, 4], keepdims=True)
  else:
    ksize = [1, 1, dropblock_size, dropblock_size]
    
    block_pattern = tf.reduce_max(-block_pattern, reduction_indices=[2])
    block_pattern = -tf.nn.max_pool(block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    block_pattern = tf.expand_dims(block_pattern, 2)

  # block_pattern = 1 - fadeoff * (1 - block_pattern)
  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)
  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  net = tf.reshape(net, [N, height, width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height, width])
  return net

def dropblock4(net, keep_prob, dropblock_size, G, data_format='channels_first'):
  """
  adaptive GN (topk-max)
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

  net_max = tf.nn.avg_pool(net, ksize=[1, 1, dropblock_size, dropblock_size], strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
  net_max = tf.pad(net_max, [[0, 0], [0, 0], [(dropblock_size - 1)//2, (dropblock_size - 1)//2], [(dropblock_size - 1)//2, (dropblock_size - 1)//2]])
  net_max = tf.reshape(net_max, [N, G, C//G, height, width]) 
  net_max = tf.transpose(net_max, [0, 1, 4, 3, 2]) 
  net_max, _ = tf.nn.top_k(net_max, k=C//G//4 if C//G > 4 else C//G, sorted=False) # [batchsize, G, width, height, (top1, top2..., topk)]
  net_max = tf.reduce_mean(net_max, axis=-1, keepdims=True)
  net_max = tf.transpose(net_max, [0, 1, 4, 3, 2]) 

  mask_max = tf.reshape(tf.nn.softmax(tf.reshape(net_max, [N, G, -1])), [N, G, 1, height, width])

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
  randnoise = tf.random_uniform([N, G, 1, height, width], dtype=tf.float32)
  # fadeoff = tf.random_uniform([N, G, 1, width, height], minval=-seed_drop_rate, maxval=seed_drop_rate, dtype=tf.float32)


  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise + mask_max) >= 1
  block_pattern = tf.cast(block_pattern, dtype=tf.float32)

  if dropblock_size == width:
    block_pattern = tf.reduce_min(block_pattern, axis=[2, 3, 4], keepdims=True)
  else:
    ksize = [1, 1, dropblock_size, dropblock_size]
    
    block_pattern = tf.reduce_max(-block_pattern, reduction_indices=[2])
    block_pattern = -tf.nn.max_pool(block_pattern, ksize=ksize, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    block_pattern = tf.expand_dims(block_pattern, 2)

  # block_pattern = 1 - fadeoff * (1 - block_pattern)
  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)
  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  net = tf.reshape(net, [N, height, width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height, width])
  return net