# -*- coding: utf-8 -*-
# File: dropblock.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import six
# from tensorpack.tfutils.compat import tfv1 as tf  # this should be avoided first in model code
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.models import GlobalAvgPooling, FullyConnected
import tensorflow as tf

__all__ = ['dropblock', 'dropblock2','dropblock3','dropblock4'] # 1: paper baseline; 2: group dropout; 3: group soft-dropout; 4: Uout group dropout  

def dropblock(net, keep_prob, dropblock_size, gap_w=None, label=None, G=None, CG=None, data_format='channels_first'):
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


def dropblock2(net, keep_prob, dropblock_size, G=None, CG=None, data_format='channels_first'):
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

  if G == None: G = C // CG
  if CG == None: CG = C // G
  net = tf.reshape(net, [N, G, CG, height, width]) 

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


def dropblock3(net, keep_prob, dropblock_size, G=None, CG=None, data_format='channels_first'):
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

  if G == None: G = C // CG
  if CG == None: CG = C // G

  net_max = tf.nn.avg_pool(net, ksize=[1, 1, dropblock_size, dropblock_size], strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
  if data_format == 'channels_last':
    _, new_height, new_width, _ = net_max.get_shape().as_list()
  else:
    _, _, new_height, new_width = net_max.get_shape().as_list()

  net_max = tf.reshape(net_max, [N, G, C//G, new_height, new_width]) 
  mask_max = tf.reduce_mean(net_max, reduction_indices=[2])
  mask_max = tf.reshape(tf.nn.softmax(tf.reshape(mask_max, [N, G, -1])), [N, G, new_height, new_width])

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
  # fadeoff = tf.random_uniform([N, G, 1, 1, 1], minval=-0.5, maxval=0.5, dtype=tf.float32)


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


def dropblock4(net, keep_prob, dropblock_size, flag=None, label=None, G=None, CG=None, data_format='channels_first'):
  '''guided drop'''
  def _get_cam(net, label, flag, dropblock_size, data_format='channels_first'):
    '''
    net: [N, C, H, W]
    gap_w : [gap_C, num_of_class]
    '''
    if data_format == 'channels_last':
      N, height, width, C = net.get_shape().as_list()
    else:
      N, C, height, width = net.get_shape().as_list()
    N = tf.shape(net)[0] 

    gap_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'linear/W') if flag > 0 else None

    if not gap_w is None:
      gap_C, num = gap_w.get_shape().as_list() # [gap_C, num]
      gap_w = tf.reshape(gap_w, [C, gap_C//C, num])
      gap_w = tf.reduce_mean(gap_w, reduction_indices=[1]) # [C, num]
      label = tf.gather(tf.transpose(gap_w), label) # [N, C]

      # spatial 
      weights = tf.expand_dims(label, 2) # [N, C, 1]
      net = tf.reshape(net, [N, height*width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height*width])
      cam = tf.matmul(weights, net, transpose_a=True) # [N, 1, width*height]
      # spt_mask = tf.not_equal(cam, tf.reduce_max(cam, reduction_indices=[2], keepdims=True))

      # cam = tf.reshape(cam, [N, height, width, 1]) if data_format == 'channels_last' else tf.reshape(cam, [N, 1, height, width])
      # cam = tf.nn.avg_pool(cam, ksize=[1, 1, dropblock_size, dropblock_size], strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
      # left_or_top = (dropblock_size-1) // 2 
      # right_or_bot = left_or_top if dropblock_size % 2 == 1 else dropblock_size-left_or_top-1
      # cam = tf.pad(cam, [[0, 0], [0, 0], [left_or_top, right_or_bot], [left_or_top, right_or_bot]])
      # cam = tf.reshape(cam, [N, height*width, 1]) if data_format == 'channels_last' else tf.reshape(cam, [N, 1, height*width])


      k = tf.cast(height*width/dropblock_size**2, tf.int32)
      topk, _ = tf.math.top_k(cam, k=k) # [N, 1, k]
      topk = tf.gather(topk, indices=[k-1], axis=-1) # [N, 1, 1]
      spt_mask = (cam < topk)

      spt_mask = tf.reshape(spt_mask, [N, height, width, 1]) if data_format == 'channels_last' else tf.reshape(spt_mask, [N, 1, height, width])

      # channel
      k = tf.cast(C/2, tf.int32)
      topk, _ = tf.math.top_k(label, k=k+1) # [N, k]
      topk = tf.gather(topk, indices=k, axis=1) # [N, 1]
      topk = tf.expand_dims(topk, 1) # [N, C, 1]
      chan_mask = (label < topk)
      chan_mask = tf.expand_dims(chan_mask, 2) # [N, C, 1]
      chan_mask = tf.expand_dims(chan_mask, 2) # [N, C, 1, 1]

      cam_mask = tf.logical_or(spt_mask, chan_mask)
      # chan_mask = tf.reshape(tf.nn.softmax(cam), [N*C, height*width]) if data_format == 'channels_last' else tf.reshape(tf.nn.softmax(cam), [N*C, height*width])
      # chan_mask = tf.reshape(cam, [N*C, height*width]) if data_format == 'channels_last' else tf.reshape(cam, [N*C, height*width])
      # chan_mask = tf.reshape(tf.nn.sigmoid(cam), [N, height, width, 1]) if data_format == 'channels_last' else tf.reshape(tf.nn.sigmoid(cam), [N, 1, height, width])
    else:
      cam_mask = False
    return cam_mask

  # def _get_gradcam(net, cost=None, gap_w=None, data_format='channels_first'):
  #   # Conv layer tensor [?,2048,10,10]
  #   def _compute_gradients(tensor, var_list):
  #     grads = tf.gradients(tensor, var_list)
  #     return [grad if grad is not None else tf.zeros_like(var)
  #             for var, grad in zip(var_list, grads)]
  #   # grads = tf.gradients(cost, net)[0]
  #   if not gap_w is None:
  #     # Normalizing the gradients
  #     if data_format == 'channels_last':
  #       N, height, width, C = net.get_shape().as_list()
  #     else:
  #       N, C, height, width = net.get_shape().as_list()
  #     N = tf.shape(net)[0] 

  #     grads = _compute_gradients(cost, [net])[0]
  #     norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads), reduction_indices=[2,3], keepdims=True)) + tf.constant(1e-5))
  #     weights = tf.reduce_mean(norm_grads, reduction_indices=[2,3])  # [N, C]
  #     weights = tf.expand_dims(weights, 2) # [N, C, 1]
  #     net = tf.reshape(net, [N, height*width, C]) if data_format == 'channels_last' else tf.reshape(net, [N, C, height*width])
  #     # cam_mean = 1 + tf.matmul(net, weights, transpose_a=True) # [N, width*height, 1]
  #     cam_mean = tf.maximum(tf.matmul(weights, net, transpose_a=True), 0) # [N, 1, width*height]
  #     cam_chan = tf.maximum(tf.multiply(net, weights), 0) # [N, C, width*height]
  #     cam = cam_mean*cam_chan
  #     # Passing through ReLU
  #     cam = cam / tf.reduce_max(cam, reduction_indices=[1,2], keepdims=True)
  #     cam = tf.reshape(cam, [N, height, width, C]) if data_format == 'channels_last' else tf.reshape(cam, [N, C, height, width])

  #   else:
  #     cam = 0.
  #   return cam

  # def _gumbel_softmax(logits, tau, shape, seed_drop_rate, eps=1e-20):
  #   if logits == False:
  #     return logits

  #   U = tf.random_uniform(tf.shape(logits), minval=0, maxval=1)
  #   y = logits - tf.log(-tf.log(U + eps) + eps)
  #   cam_mask = tf.nn.softmax(y / tau)
  #   topk, _ = tf.math.top_k(cam_mask, k=tf.cast(seed_drop_rate*shape[-1], tf.int32)) # [N, 1]
  #   topk = tf.gather(topk, indices=tf.cast(seed_drop_rate*shape[-1], tf.int32)-1, axis=1)
  #   topk = tf.expand_dims(topk, 1) # [N, C, 1]
  #   cam_mask = (cam_mask < topk)
  #   # cam_mask = tf.cast(tf.equal(cam_mask, tf.reduce_max(cam_mask, reduction_indices=[1], keepdims=True)), tf.float32)

  #   cam_mask = tf.expand_dims(cam_mask, 2) # [N, C, 1]
  #   cam_mask = tf.expand_dims(cam_mask, 2) # [N, C, 1, 1]
  #   return cam_mask


  ctx = get_current_tower_context()
  is_training = bool(ctx.is_training)

  if not is_training or keep_prob is None:
    return net

  tf.logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(dropblock_size, net.shape))

  if data_format == 'channels_last':
    _, width, height, C = net.get_shape().as_list()
  else:
    _, C, width, height = net.get_shape().as_list()
  if width != height:
    raise ValueError('Input tensor with width!=height is not supported.')
  N = tf.shape(net)[0] 

  dropblock_size = min(dropblock_size, width)
  # seed_drop_rate is the gamma parameter of DropBlcok.
  seed_drop_rate = (1.0 - keep_prob) * width**2 / dropblock_size**2 / (width - dropblock_size + 1)**2

  cam_mask = _get_cam(net, label, flag, dropblock_size, data_format)

  # Forces the block to be inside the feature map.
  w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
  valid_block_center = tf.logical_and(
      tf.logical_and(w_i >= int(dropblock_size // 2),
                     w_i < width - (dropblock_size - 1) // 2),
      tf.logical_and(h_i >= int(dropblock_size // 2),
                     h_i < width - (dropblock_size - 1) // 2))

  valid_block_center = tf.expand_dims(valid_block_center, 0)
  valid_block_center = tf.expand_dims(valid_block_center, -1 if data_format == 'channels_last' else 0)

  randnoise = tf.random_uniform(tf.shape(net), dtype=tf.float32)
  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast((1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1
  block_pattern = tf.logical_or(block_pattern, cam_mask)
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

  percent_ones = tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)

  net = net / tf.cast(percent_ones, net.dtype) * tf.cast(block_pattern, net.dtype)
  return net


def dropblock10(net, keep_prob, dropblock_size, G=None, CG=None, data_format='channels_first'):
  """
  adaptive GN (nonlocal)
  """  
  def zca(features, shape, eps=1e-1):
    # [N,G,C,H,W] -> [N,G,C*H*W]
    mean = tf.reduce_mean(features, axis=[2, 3, 4], keepdims=True)
    features = features-mean
    unbiased_features = tf.reshape(features-mean, [shape[0], shape[1], -1])

    # get the covariance matrix
    # [N,G,C*H*W] -> [N,G,G]
    # gram = tf.matmul(unbiased_features, unbiased_features, transpose_a=True)
    gram = tf.einsum('nis,njs->nij', unbiased_features, unbiased_features)
    gram = gram / (tf.cast(shape[2]*shape[3]*shape[4], tf.float32) - 1.) + tf.eye(shape[1])*eps
    # gram /= tf.reduce_prod(tf.cast(shape[2:4], tf.float32))

    # converting the feature spaces
    with tf.device('/cpu:0'): 
      S, U, V = tf.svd(gram, compute_uv=True) # S: [N, G]  U: [N, G, G]

    # valid_index = tf.reduce_sum(tf.cast(tf.greater(S, 1e-5), tf.int32))
    S = tf.expand_dims(S, axis=1)
    valid_index = tf.cast(S > 1e-5, dtype=tf.float32)
    S_filted = tf.sqrt(1.0/tf.maximum(S, 1e-5)) * valid_index

    normalized = tf.matmul(unbiased_features, U,  transpose_a=True)
    normalized = tf.multiply(normalized, S_filted) # [N, HWC, G]
    normalized = tf.matmul(normalized, V, transpose_b=True)

    normalized = tf.reshape(normalized, shape=shape)
    # # Whiten
    # S_whitened = tf.diag(tf.pow(S[:,:valid_index], -0.5))
    # # S x U x S^T x X
    # normalized = tf.matmul(U[:,:,:valid_index], S_whitened)
    # normalized = tf.matmul(normalized, U[:,:,:valid_index], transpose_b=True)
    # normalized = tf.matmul(normalized, unbiased_features)
    # normalized = tf.reshape(normalized, shape=shape)

    return normalized

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

  if G == None: G = C // CG
  if CG == None: CG = C // G
  net = tf.reshape(net, [N, G, CG, height, width]) 

  net_max = zca(net, [N, G, CG, height, width])
  mask_max = tf.reduce_mean(net_max, reduction_indices=[2], keepdims=True)
  mask_max = tf.nn.sigmoid(mask_max)
  # mask_max = tf.reshape(tf.nn.softmax(tf.reshape(mask_max, [N, G, -1])), [N, G, 1, height, width])

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

  block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) + tf.cast(
      (1 - seed_drop_rate), dtype=tf.float32) + randnoise * mask_max) >= 1
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
