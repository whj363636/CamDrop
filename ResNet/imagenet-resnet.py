#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
import os
import tensorflow as tf
import numpy as np
import os
import sys
import cv2


from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils import viz
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, fbresnet_augmentor
from resnet_model import preresnet_basicblock, preresnet_group

from tensorpack import QueueInput, TFDatasetInput, logger
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.train import SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, eval_on_ILSVRC12, get_imagenet_dataflow, get_imagenet_tfdata
from resnet_model import resnet_basicblock, preresnet_basicblock, preresnet_bottleneck, preresnet_group, se_resnet_bottleneck


parser = argparse.ArgumentParser()
# generic:
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
parser.add_argument('--load', help='load a model for training or evaluation')
# parser.add_argument('--seed', default=1234, type=int, help="seed")

# data:
parser.add_argument('--data', help='ILSVRC dataset dir')
parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
parser.add_argument('--symbolic', help='use symbolic data loader', action='store_true')

# model:
parser.add_argument('--data-format', help='the image data layout used by the model', default='NCHW', choices=['NCHW', 'NHWC'])
parser.add_argument('-d', '--depth', help='ResNet depth', type=int, default=50, choices=[18, 34, 50, 101, 152])
parser.add_argument('--weight-decay-norm', action='store_true', help="apply weight decay on normalization layers (gamma & beta)." "This is used in torch/pytorch, and slightly " "improves validation accuracy of large models.")
parser.add_argument('--batch', default=256, type=int, help="total batch size. " "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy." "Pretrained models listed in README were trained with batch=32x8.")
parser.add_argument('--mode', choices=['resnet', 'preact', 'se'], help='variants of resnet to use', default='resnet')

parser.add_argument('--lrs', nargs='+', default=[50, 80, 100, 105, 270], type=int, help='or [60, 120, 180, 190, 200]')
parser.add_argument('--start', type=int, default=1, help='The start epoch.')
parser.add_argument('--keep_prob', type=float, default=None, help='The keep probabiltiy of dropblock.')
parser.add_argument('--blocksize', type=int, default=7, help='The size of dropblock.')
parser.add_argument('--groupsize', type=int, default=64, help='The size of groupdrop.')
parser.add_argument('--dropblock_groups', type=str, default='3,4', help='which group to drop')
parser.add_argument('--norm', type=str, default='BN', help='BN or GN')
parser.add_argument('--strategy', type=str, default=None, help='strategy for dropblock, decay or not')
parser.add_argument('--ablation', type=str, default='', help='.')
parser.add_argument('--cam', action='store_true', help='run visualization')

args = parser.parse_args()

if args.norm == 'BN':
    from resnet_model import resnet_backbone, resnet_bottleneck, resnet_group
elif args.norm == 'GN':
    from resnet_model_GN import resnet_backbone, resnet_bottleneck, resnet_group


class Model(ImageNetModel):
    def __init__(self, depth, keep_probs, mode='resnet'):
        if mode == 'se':
            assert depth >= 50

        self.keep_probs = keep_probs
        self.mode = mode
        self.flag = -1

        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image, label):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(image, label, self.num_blocks, preresnet_group if self.mode == 'preact' else resnet_group, self.block_func, self.flag, args)


def get_config(model):
    nr_tower = max(get_num_gpu(), 1)
    assert args.batch % nr_tower == 0
    batch = args.batch // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    if batch < 32 or batch > 64:
        logger.warn("Batch size per tower not in [32, 64]. This probably will lead to worse accuracy than reported.")
    if args.fake:
        data = QueueInput(FakeData(
            [[batch, 224, 224, 3], [batch]], 1000, random=False, dtype='uint8'))
        callbacks = []
    else:
        if args.symbolic:
            data = TFDatasetInput(get_imagenet_tfdata(args.data, 'train', batch))
        else:
            data = QueueInput(get_imagenet_dataflow(args.data, 'train', batch))

        START_LR = 0.1
        BASE_LR = START_LR * (args.batch / 256.0)
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),

            ScheduledHyperParamSetter(
                'learning_rate', [ # 90-144-190-195-200 / 30-60-90-100-105
                    (0, min(START_LR, BASE_LR)), (args.lrs[0], BASE_LR * 1e-1), (args.lrs[1], BASE_LR * 1e-2),
                    (args.lrs[2], BASE_LR * 1e-3), (args.lrs[3], BASE_LR * 1e-4)]),
        ]
        if BASE_LR > START_LR:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, START_LR), (5, BASE_LR)], interp='linear'))

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        dataset_val = get_imagenet_dataflow(args.data, 'val', batch)
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    if get_num_gpu() > 0:
        callbacks.append(GPUUtilizationTracker())

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1281167 // args.batch,
        starting_epoch=args.start,
        max_epoch=args.lrs[4],
    )


def viz_cam(model_file, data_dir):
    def get_data(train_or_test):
        # completely copied from imagenet-resnet.py example
        isTrain = train_or_test == 'train'

        datadir = args.data
        ds = dataset.ILSVRC12(datadir, train_or_test, shuffle=isTrain)
        augmentors = fbresnet_augmentor(isTrain)
        augmentors.append(imgaug.ToUint8())

        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if isTrain:
            ds = PrefetchDataZMQ(ds, min(25, multiprocessing.cpu_count()))
        ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
        return ds

    ds = get_data('val')
    pred_config = PredictConfig(
        model=Model(args.depth, args.keep_prob, args.mode),
        session_init=get_model_loader(model_file),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'group3/block2/conv3/output', 'linear/W'],
        return_input=True
    )
    meta = dataset.ILSVRCMeta().get_synset_words_1000()

    pred = SimpleDatasetPredictor(pred_config, ds)
    cnt = 0
    for inp, outp in pred.get_result():
        images, labels = inp
        wrongs, convmaps, W = outp
        batch = wrongs.shape[0]
        for i in range(batch):
            if wrongs[i]:
                continue
            weight = W[:, [labels[i]]].T    # 512x1
            convmap = convmaps[i, :, :, :]  # 512xhxw
            mergedmap = np.matmul(weight, convmap.reshape((2048, -1))).reshape(7, 7)
            mergedmap = cv2.resize(mergedmap, (224, 224))
            heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
            blend = images[i] * 0.5 + heatmap * 0.5
            concat = np.concatenate((images[i], heatmap, blend), axis=1)

            classname = meta[labels[i]].split(',')[0]
            if not os.path.exists('cam_db'):
                os.makedirs('cam_db')
            cv2.imwrite(os.path.join('cam_db', 'cam{}-{}.jpg'.format(cnt, classname)), concat)
            cnt += 1

if __name__ == '__main__':
    # tf.random.set_random_seed(args.seed)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.keep_prob, args.mode)
    model.data_format = args.data_format
    if args.weight_decay_norm:
        model.weight_decay_pattern = ".*/W|.*/gamma|.*/beta"
    if args.cam:
        BATCH_SIZE = 128    # something that can run on one gpu
        viz_cam(args.load, args.data)
        sys.exit()
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_imagenet_dataflow(args.data, 'val', batch)
        if args.atk_type:
            attacker = FastGradientMethod(model)
            x_adv = attacker.generate(x_input, eps=args.eps, clip_min=-1., clip_max=1.) 
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.fake:
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            logger.set_logger_dir(
                os.path.join('train_log',
                             'imagenet-batch{}-norm{}-drop{}-groups{}-groupsize{}-{}'.format(
                                 args.batch, args.norm, args.keep_prob, args.dropblock_groups, args.groupsize, args.ablation)))

        config = get_config(model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
        launch_train_with_config(config, trainer)
