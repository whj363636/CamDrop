# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

from tensorpack.models import regularize_cost, BatchNorm
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import get_current_tower_context, TowerFuncWrapper
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once
from tensorpack.tfutils.collection import freeze_collection
from tensorpack.tfutils.varreplace import custom_getter_scope

from third_party.imagenet_utils import ImageNetModel


IMAGE_SCALE = 2.0 / 255


class NoOpAttacker():
    """
    A placeholder attacker which does nothing.
    """
    def attack(self, image, label, model_func):
        return image, -tf.ones_like(label)


"""The FastGradientMethod attack."""
class FGSMAttacker():
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method. Paper link: https://arxiv.org/abs/1412.6572
    """
    def __init__(self, num_iter, epsilon, clip_min=-1., clip_max=1.):
        self.num_iter = num_iter
        # rescale the attack epsilon and attack step size
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def _create_random_target(self, label):
        label_offset = tf.random_uniform(tf.shape(label), minval=1, maxval=1000, dtype=tf.int32)
        return tf.floormod(label + label_offset, tf.constant(1000, tf.int32))

    def attack(self, image_clean, label, model_func):
        logits = model_func(image_clean, label)
        logits = tf.cast(logits, tf.float32)
        target = self._create_random_target(label)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target)

        x_adv = image_clean + self.epsilon*tf.sign(tf.gradients(loss, image_clean)[0])
        x_adv = tf.clip_by_value(x_adv, self.clip_min, self.clip_max)

        return tf.stop_gradient(x_adv), target


class PGDAttacker():
    """
    The Projected Gradient Descent attack is a variant of the Basic Iterative Method in which,
    after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted
    data range). This is the attack proposed by Madry et al. for adversarial training.
    Paper link: https://arxiv.org/abs/1706.06083
    """
    USE_FP16 = False
    """
    Use FP16 to run PGD iterations.
    This has about 2~3x speedup for most types of models
    if used together with XLA on Volta GPUs.
    """

    USE_XLA = False
    """
    Use XLA to optimize the graph of PGD iterations.
    This requires CUDA>=9.2 and TF>=1.12.
    """

    def __init__(self, num_iter, epsilon, step_size, prob_start_from_clean=0.0):
        """
        Args:
            num_iter (int):
            epsilon (float):
            step_size (int):
            prob_start_from_clean (float): The probability to initialize with
                the original image, rather than a randomly perturbed one.
        """
        step_size = max(step_size, epsilon / num_iter)
        """
        Feature Denoising, Sec 6.1:
        We set its step size α = 1, except for 10-iteration attacks where α is set to ε/10=1.6
        """
        self.num_iter = num_iter
        # rescale the attack epsilon and attack step size
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size * IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean

    def _create_random_target(self, label):
        """
        Feature Denoising Sec 6:
        we consider targeted attacks when
        evaluating under the white-box settings, where the targeted
        class is selected uniformly at random
        """
        label_offset = tf.random_uniform(tf.shape(label), minval=1, maxval=1000, dtype=tf.int32)
        return tf.floormod(label + label_offset, tf.constant(1000, tf.int32))

    def attack(self, image_clean, label, model_func):
        target_label = self._create_random_target(label)

        def fp16_getter(getter, *args, **kwargs):
            name = args[0] if len(args) else kwargs['name']
            if not name.endswith('/W') and not name.endswith('/b'):
                """
                Following convention, convolution & fc are quantized.
                BatchNorm (gamma & beta) are not quantized.
                """
                return getter(*args, **kwargs)
            else:
                if kwargs['dtype'] == tf.float16:
                    kwargs['dtype'] = tf.float32
                    ret = getter(*args, **kwargs)
                    ret = tf.cast(ret, tf.float16)
                    log_once("Variable {} casted to fp16 ...".format(name))
                    return ret
                else:
                    return getter(*args, **kwargs)

        def one_step_attack(adv):
            if not self.USE_FP16:
                logits = model_func(adv, label)
            else:
                adv16 = tf.cast(adv, tf.float16)
                with custom_getter_scope(fp16_getter):
                    logits = model_func(adv16, label)
                    logits = tf.cast(logits, tf.float32)
            # Note we don't add any summaries here when creating losses, because
            # summaries don't work in conditionals.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target_label)  # we want to minimize it in targeted attack
            if not self.USE_FP16:
                g, = tf.gradients(losses, adv)
            else:
                """
                We perform loss scaling to prevent underflow:
                https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                (We have not yet tried training without scaling)
                """
                g, = tf.gradients(losses * 128., adv)
                g = g / 128.

            """
            Feature Denoising, Sec 5:
            We use the Projected Gradient Descent (PGD)
            (implemented at https://github.com/MadryLab/cifar10_challenge )
            as the white-box attacker for adversarial training
            """
            adv = tf.clip_by_value(adv - tf.sign(g) * self.step_size, lower_bound, upper_bound)
            return adv

        """
        Feature Denoising, Sec 6:
        Adversarial perturbation is considered under L∞ norm (i.e., maximum difference for each pixel).
        """
        lower_bound = tf.clip_by_value(image_clean - self.epsilon, -1., 1.)
        upper_bound = tf.clip_by_value(image_clean + self.epsilon, -1., 1.)

        """
        Feature Denoising Sec. 5:
        We randomly choose from both initializations in the
        PGD attacker during adversarial training: 20% of training
        batches use clean images to initialize PGD, and 80% use
        random points within the allowed .
        """
        init_start = tf.random_uniform(tf.shape(image_clean), minval=-self.epsilon, maxval=self.epsilon)
        start_from_noise_index = tf.cast(tf.greater(
            tf.random_uniform(shape=[]), self.prob_start_from_clean), tf.float32)
        start_adv = image_clean + start_from_noise_index * init_start

        if self.USE_XLA:
            assert tuple(map(int, tf.__version__.split('.')[:2])) >= (1, 12)
            from tensorflow.contrib.compiler import xla
        with tf.name_scope('attack_loop'):
            adv_final = tf.while_loop(
                lambda _: True,
                one_step_attack if not self.USE_XLA else
                lambda adv: xla.compile(lambda: one_step_attack(adv))[0],
                [start_adv],
                back_prop=False,
                maximum_iterations=self.num_iter,
                parallel_iterations=1)
        return adv_final, target_label


class AdvImageNetModel(ImageNetModel):

    """
    Feature Denoising, Sec 5:
    A label smoothing of 0.1 is used.
    """
    label_smoothing = 0.1

    def set_attacker(self, attacker):
        self.attacker = attacker

    def build_graph(self, image, label):
        """
        The default tower function.
        """
        image = self.image_preprocess(image)
        assert self.data_format == 'NCHW'
        image = tf.transpose(image, [0, 3, 1, 2])
        ctx = get_current_tower_context()

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            # BatchNorm always comes with trouble. We use the testing mode of it during attack.
            with freeze_collection([tf.GraphKeys.UPDATE_OPS]), argscope(BatchNorm, training=False):
                image, target_label = self.attacker.attack(image, label, self.get_logits)
                image = tf.stop_gradient(image, name='adv_training_sample')

            logits = self.get_logits(image, label)

        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)
        AdvImageNetModel.compute_attack_success(logits, target_label)
        if not ctx.is_training:
            return

        wd_loss = regularize_cost(self.weight_decay_pattern,
                                  tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        total_cost = tf.add_n([loss, wd_loss], name='cost')

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    def get_inference_func(self, attacker):
        """
        Returns a tower function to be used for inference. It generates adv
        images with the given attacker and runs classification on it.
        """

        def tower_func(image, label):
            assert not get_current_tower_context().is_training
            image = self.image_preprocess(image)
            image = tf.transpose(image, [0, 3, 1, 2])
            image, target_label = attacker.attack(image, label, self.get_logits)
            logits = self.get_logits(image)
            ImageNetModel.compute_loss_and_error(logits, label)  # compute top-1 and top-5
            AdvImageNetModel.compute_attack_success(logits, target_label)

        return TowerFuncWrapper(tower_func, self.get_inputs_desc())

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            # For the purpose of adversarial training, normalize images to [-1, 1]
            image = image * IMAGE_SCALE - 1.0
            return image

    @staticmethod
    def compute_attack_success(logits, target_label):
        """
        Compute the attack success rate.
        """
        pred = tf.argmax(logits, axis=1, output_type=tf.int32)
        equal_target = tf.equal(pred, target_label)
        success = tf.cast(equal_target, tf.float32, name='attack_success')
        add_moving_summary(tf.reduce_mean(success, name='attack_success_rate'))
