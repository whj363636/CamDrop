#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import numpy as np
import os
import socket
import sys

import horovod.tensorflow as hvd

from tensorpack import *
from tensorpack.tfutils import get_model_loader

from adv_model import NoOpAttacker, PGDAttacker, FGSMAttacker, AdvImageNetModel
from third_party.imagenet_utils import get_val_dataflow, eval_on_ILSVRC12
from third_party.utils import HorovodClassificationError
sys.path.append("..")
from resnet_model import resnet_basicblock, preresnet_basicblock, preresnet_bottleneck, preresnet_group, se_resnet_bottleneck


parser = argparse.ArgumentParser()
# run on a directory of images:
parser.add_argument('--eval-directory', help='Path to a directory of images to classify.')
parser.add_argument('--prediction-file', help='Path to a txt file to write predictions.', default='predictions.txt')

parser.add_argument('--data', help='ILSVRC dataset dir')
parser.add_argument('--fake', help='Use fakedata to test or benchmark this model', action='store_true')
parser.add_argument('--no-zmq-ops', help='Use pure python to send/receive data', action='store_true')
parser.add_argument('--batch', help='Per-GPU batch size', default=32, type=int)

# architecture flags:
parser.add_argument('-d', '--depth', help='ResNet depth',type=int, default=50, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py', default='resnet')
parser.add_argument('--groupsize', type=int, default=64, help='The size of groupdrop.')
parser.add_argument('--dropblock_groups', type=str, default=None, help='which group to drop')
parser.add_argument('--use-fp16xla', help='Optimize PGD with fp16+XLA in training or evaluation. ''(Evaluation during training will still use FP32, for fair comparison)', action='store_true')

# attacker flags:
parser.add_argument('--load', help='Path to a model to load for evaluation or resuming training.')
parser.add_argument('--eval', action='store_true', help='Evaluate a model on ImageNet instead of training.')

parser.add_argument('--norm', type=str, default='BN', help='BN or GN')
parser.add_argument('--ak_type', type=str, default='PGD', help='PGD or FGSM')
parser.add_argument('--iter', help='Adversarial attack iteration',type=int, default=2)
parser.add_argument('--epsilon', help='Adversarial attack maximal perturbation',type=float, default=5.0)
parser.add_argument('--stepsize', help='Adversarial attack step size',type=float, default=1.0)


args = parser.parse_args()

if args.norm == 'BN':
    from resnet_model import resnet_backbone, resnet_bottleneck, resnet_group
elif args.norm == 'GN':
    from resnet_model_GN import resnet_backbone, resnet_bottleneck, resnet_group

class Model(AdvImageNetModel):
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


def create_eval_callback(name, tower_func, condition):
    """
    Create a distributed evaluation callback.

    Args:
        name (str): a prefix
        tower_func (TowerFuncWrapper): the inference tower function
        condition: a function(epoch number) that returns whether this epoch should evaluate or not
    """
    dataflow = get_val_dataflow(
        args.data, args.batch,
        num_splits=hvd.size(), split_index=hvd.rank())
    # We eval both the classification error rate (for comparison with defenders)
    # and the attack success rate (for comparison with attackers).
    infs = [HorovodClassificationError('wrong-top1', '{}-top1-error'.format(name)),
            HorovodClassificationError('wrong-top5', '{}-top5-error'.format(name)),
            HorovodClassificationError('attack_success', '{}-attack-success-rate'.format(name))
            ]
    cb = InferenceRunner(
        QueueInput(dataflow), infs,
        tower_name=name,
        tower_func=tower_func).set_chief_only(False)
    cb = EnableCallbackIf(
        cb, lambda self: condition(self.epoch_num))
    return cb


if __name__ == '__main__':
    # Define model
    model = Model(args.depth, keep_probs=None, mode=args.arch)

    # Define attacker
    if args.iter == 0 or args.eval_directory:
        attacker = NoOpAttacker()
    else:
        if args.ak_type == 'PGD':
            attacker = PGDAttacker(args.iter, args.epsilon, args.stepsize, prob_start_from_clean=0.2 if not args.eval else 0.0)
            if args.use_fp16xla:
                attacker.USE_FP16 = True
                attacker.USE_XLA = True
        elif args.ak_type == 'FGSM':
            attacker = FGSMAttacker(1, args.epsilon)

    model.set_attacker(attacker)
    hvd.init()

    if args.eval:
        sessinit = get_model_loader(args.load)
        if hvd.size() == 1:
            # single-GPU eval, slow
            ds = get_val_dataflow(args.data, args.batch)
            eval_on_ILSVRC12(model, sessinit, ds)
        else:
            logger.info("CMD: " + " ".join(sys.argv))
            cb = create_eval_callback(
                "eval",
                model.get_inference_func(attacker),
                lambda e: True)
            trainer = HorovodTrainer()
            trainer.setup_graph(model.get_inputs_desc(), PlaceholderInput(), model.build_graph, model.get_optimizer)
            # train for an empty epoch, to reuse the distributed evaluation code
            trainer.train_with_defaults(
                callbacks=[cb],
                monitors=[ScalarPrinter()] if hvd.rank() == 0 else [],
                session_init=sessinit,
                steps_per_epoch=0, max_epoch=1)

    elif args.eval_directory:
        assert hvd.size() == 1
        files = glob.glob(os.path.join(args.eval_directory, '*.*'))
        ds = ImageFromFile(files)
        # Our model expects BGR images instead of RGB.
        # Also do a naive resize to 224.
        ds = MapData(
            ds,
            lambda dp: [cv2.resize(dp[0][:, :, ::-1], (224, 224), interpolation=cv2.INTER_CUBIC)])
        ds = BatchData(ds, args.batch, remainder=True)

        pred_config = PredictConfig(
            model=model,
            session_init=get_model_loader(args.load),
            input_names=['input'],
            output_names=['linear/output']  # the logits
        )
        predictor = SimpleDatasetPredictor(pred_config, ds)

        logger.info("Running inference on {} images in {}".format(len(files), args.eval_directory))
        results = []
        for logits, in predictor.get_result():
            predictions = list(np.argmax(logits, axis=1))
            results.extend(predictions)
        assert len(results) == len(files)
        with open(args.prediction_file, "w") as f:
            for filename, label in zip(files, results):
                f.write("{},{}\n".format(filename, label))
        logger.info("Outputs saved to " + args.prediction_file)
