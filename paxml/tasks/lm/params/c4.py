# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Model configurations on the T5/C4 dataset."""

import functools
import math
from typing import Dict, List, Optional

from absl import logging
import fiddle as fdl
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm import model_params
# from paxml.tasks.lm.params import lm_cloud  # XD
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import normalizations  # XD
from praxis.layers import transformers
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
from paxml.tasks.lm.params import global_cfg  # XD

WeightInit = base_layer.WeightInit

GPT_SPM_PATH = global_cfg.GPT_SPM_PATH #(  XD
#     'gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model'  # XD
#     # 'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
# )
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}
C4_GPT_EVAL_FEATURES_LM = {
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}
C4_TRAIN_DATADIR = global_cfg.C4_TRAIN_DATADIR # XD 'gs://common_datasets'  # XD: 'gs://mlperf-llm-public2'
C4_EVAL_DATADIR = global_cfg.C4_EVAL_DATADIR # XD 'gs://common_datasets' # XD: 'gs://mlperf-llm-public2'

# XD
# import tensorflow as tf
# RT_DATAPATH = 'gs://llm_projects/data/rotten_tomatoes_train_8530.tfrecords'
# feature_desc = {"input_ids": tf.io.VarLenFeature(tf.int64)}
# RT_GPT_FEATURES_LM = {'targets': seqio.Feature(vocabulary=PASS_THROUGH_VOCABULARY, dtype=tf.int32, rank=1)}

# @seqio.map_over_dataset
# def convert_datatype(ex):
#   return {k: tf.cast(tf.sparse.to_dense(v, default_value=0), dtype=tf.int32) for k, v in ex.items()}

# seqio.TaskRegistry.add('rotten_tomatoes_lm_gpt', 
#     seqio.TFExampleDataSource(split_to_filepattern={'train': RT_DATAPATH}, feature_description=feature_desc),
#     preprocessors=[
#         convert_datatype,
#         functools.partial(t5_preprocessors.rekey, key_map={'targets': 'input_ids'}),
#     ],
#     output_features=RT_GPT_FEATURES_LM,
# )

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks


# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt',
    versions=['3.0.1'],  # XD: 3.0.4 -> 3.0.1
    pinned_version='3.0.1',  # XD: 3.0.4 -> 3.0.1
    tfds_name='c4/en',
    tfds_data_dir=C4_TRAIN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            },
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            t5_preprocessors.reduce_concat_tokens,
            batch_size=4096,
        ),
        t5_preprocessors.split_tokens_to_targets_length,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt_eval_tokenized',
    versions=['3.0.1'],  # XD: 3.0.5 -> 3.0.1
    pinned_version='3.0.1',  # XD: 3.0.5 -> 3.0.1
    tfds_name='c4/en',
    tfds_data_dir=C4_EVAL_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'ids',
            },
        ),
        seqio.preprocessors.tokenize,
    ],
    output_features=C4_GPT_EVAL_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=None,
)


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  PERCORE_EVAL_BATCH_SIZE = None
  MAX_SEQ_LEN = 1024
  TRAINING_SEED = 9876
  TRAINING_NUM_BATCHES_TO_SKIP = None

  def _dataset_common(
      self, is_training
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    if is_training:
      percore_batch_size = self.PERCORE_BATCH_SIZE
    else:
      if self.PERCORE_EVAL_BATCH_SIZE is not None:
        percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
      else:
        percore_batch_size = self.PERCORE_BATCH_SIZE

    num_local_devices = jax.local_device_count()
    global_batch_size = int(
        percore_batch_size * num_local_devices * jax.process_count() + 1e-6
    )
    if percore_batch_size >= 1:
      assert global_batch_size % num_local_devices == 0
      batch_size_per_process = int(
          math.ceil(percore_batch_size) * num_local_devices + 1e-6
      )
      num_infeed_hosts = global_batch_size // batch_size_per_process
    else:
      if jax.process_count() > 1:
        # assert global_batch_size % num_local_devices == 0  # XD: bug?
        # batch_size_per_process = num_local_devices  # XD: bug?
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    # batch_size_per_process, num_infeed_hosts = 4, 2  # XD
    seed = None
    if is_training:
      seed = self.TRAINING_SEED
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        name='C4Train' if is_training else 'C4Validation',
        mixture_name='c4_lm_v301_gpt'
        if is_training
        else 'c4_lm_v301_gpt_eval_tokenized',
        # split_name='train2' if is_training else 'validation_tokenized_5662seqs',
        split_name='train' if is_training else 'validation',  # XD for 3.0.1
        # mixture_name='rotten_tomatoes_lm_gpt', split_name='train',  # XD
        task_feature_lengths={'targets': self.MAX_SEQ_LEN},
        use_cached=False,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=True if is_training else False,
            use_custom_packing_ops=False,
            bos_id=0,
            reverse_bos_padding=True,
            eos_id=GPT_EOS_ID,
        ),
        is_training=is_training,
        input_random_seed=(seed if is_training else 4321),
        batch_size=batch_size_per_process,
        drop_remainder=True if is_training else False,
        num_batches_to_skip=self.TRAINING_NUM_BATCHES_TO_SKIP,
        num_infeed_hosts=num_infeed_hosts,
        reset_for_eval=False if is_training else True,
        annotate_padding_fields=True,
    )
    return p

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


def set_adam_and_learning_rate_schedule(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Sets the Adam optimizer and the learning rate schedule."""
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = pax_fiddle.Config(
      optimizers.Adam,
      beta1=cls.ADAM_BETA1 if cls.ADAM_BETA1 else 0.9,
      beta2=cls.ADAM_BETA2 if cls.ADAM_BETA2 else 0.999,
      weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
      epsilon=cls.ADAM_EPSILON if cls.ADAM_EPSILON else 1e-6,
      epsilon_root=cls.ADAM_EPSILON_ROOT if cls.ADAM_EPSILON_ROOT else 0.0,
      clip_gradient_norm_to_value=cls.CLIP_GRADIENT_NORM_TO_VALUE
      if cls.CLIP_GRADIENT_NORM_TO_VALUE
      else 5.0,
      clip_threshold=cls.ADAM_CLIP_THRESHOLD
      if cls.ADAM_CLIP_THRESHOLD
      else 1.0,
  )

  if hasattr(cls, 'PERCORE_BATCH_SIZE'):
    global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
    if global_batch_size == 0:
      logging.warning(
          (
              'Found global_batch_size = 0: cls.PERCORE_BATCH_SIZE=%s,'
              ' jax.device_count()=%s'
          ),
          cls.PERCORE_BATCH_SIZE,
          jax.device_count(),
      )
    assert global_batch_size <= 8192
  else:
    global_batch_size = None

  if cls.LEARNING_RATE is not None:
    lp.optimizer.learning_rate = cls.LEARNING_RATE
  else:
    assert global_batch_size is not None
    if global_batch_size <= 3584:
      lp.optimizer.learning_rate = 2e-5
    else:
      lp.optimizer.learning_rate = 3e-5

  if cls.LR_SCHEDULE == 'linear_rampup_exponential_decay':
    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupExponentialDecay,
        warmup_steps=cls.LR_LRED_WARMUP,
        decay_start=cls.LR_LRED_DECAY_START,
        decay_end=cls.LR_LRED_DECAY_END,
        min_ratio=cls.LR_LRED_MIN_RATIO,
        max=cls.LR_LRED_MAX,
    )
  elif cls.LR_SCHEDULE == 'linear_rampup_cosine_decay':
    if cls.LR_COS_WARMUP is not None:
      warmup_steps = cls.LR_COS_WARMUP
    else:
      assert global_batch_size is not None
      warmup_steps = math.ceil(265.0 * 1536 / global_batch_size - 1e-6)
      assert warmup_steps > 0

    if cls.LR_COS_DECAY_START is not None:
      decay_start_step = cls.LR_COS_DECAY_START
    else:
      decay_start_step = warmup_steps + 1

    if cls.LR_COS_DECAY_END is not None:
      decay_end_step = cls.LR_COS_DECAY_END
    else:
      assert global_batch_size is not None
      decay_end_step = math.ceil(108600.0 * 1536 / global_batch_size - 1e-6)
      assert decay_end_step > 0

    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=warmup_steps,
        decay_start=decay_start_step,
        decay_end=decay_end_step,
        min_ratio=cls.LR_COS_MIN_RATIO,
        max=cls.LR_COS_MAX,
    )
  else:
    raise NotImplementedError(
        f'Learning rate schedule {cls.LR_SCHEDULE} is not supported.'
    )

  return task_p


class TransformerLmSpmdAdam(model_params.TransformerLmSpmdAdafactor):
  """Base SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 2e-4  # XD: 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95  # XD: 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-8  # XD: -6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'  # XD: exponential
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  # XD
  LR_COS_WARMUP = 2000
  LR_COS_DECAY_START = 2001
  LR_COS_DECAY_END = 200000
  # LR_COS_WARMUP = 4000
  # LR_COS_DECAY_START = 4001
  # LR_COS_DECAY_END = 300000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class TransformerLmSpmdPipelineAdam(
    model_params.TransformerLmSpmdPipelineAdafactor
):
  """Base pipelined SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdPipelineAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam): # XD, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""
  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


@experiment_registry.register
class LmCloudSpmdAdamLimitSteps(LmCloudSpmdAdam):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 4000
    return task_p


class EarlyStoppingFn(base_hyperparams.FiddleBaseParameterizable):
  r"""Early stopping function to log eval log_pplx and stop when reaching target.

  Attributes:
    target_log_pplx: target log pplx value to stop training when eval log pplx
      reaches this value.
  """

  target_log_pplx: Optional[float] = None

  def __call__(
      self,
      metrics: Dict[str, float],
      running_mode: trainer_lib.RunningMode,
      global_step: int,
      is_last_ckpt: bool,
  ) -> bool:
    """Returns True if run should be stopped early."""
    if 'eval_test_C4Validation/metrics/log_pplx' not in metrics.keys():
      return False
    log_pplx = metrics['eval_test_C4Validation/metrics/log_pplx']

    if log_pplx <= self.target_log_pplx:
      return True
    return False


def configure_gpt3_task(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Returns task with gpt3 related configs."""
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.params_init = WeightInit.Gaussian(0.006)

  softmax_init = WeightInit.Gaussian(0.006)
  model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None

  if cls.SEPARATE_EMBEDDING:
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (
        cls.EMBEDDING_LOOKUP_STYLE
    )
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  stacked_p = model_p.lm_tpl.stacked_transformer_tpl
  if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(
      fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
  ):
    stacked_p = stacked_p.block
  transformer_layer_p = stacked_p.transformer_layer_params_tpl

  transformer_layer_p.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  transformer_layer_p.tr_fflayer_tpl.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  model_p.lm_tpl.final_ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  if False and cls.NORMALIZATION_CLS == normalizations.RmsNorm:  # XD
    transformer_layer_p.ln_tpl.intermediate_dtype = jnp.float32
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.intermediate_dtype = jnp.float32
    model_p.lm_tpl.final_ln_tpl.intermediate_dtype = jnp.float32
  if cls.NORMALIZATION_CLS == normalizations.LayerNorm:  # XD
    transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
    model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
  transformer_layer_p.tr_atten_tpl.use_bias = cls.USE_BIAS  # XD: True
  # XD
  for name in ['num_groups', 'project_logits', 'project_probs', 
              'logits_residual', 'probs_residual', 'logits_absorb_residual', 'probs_absorb_residual',
              'logits_squeeze_ratio', 'logits_squeeze_activation_cls', 'logits_output_activation_cls',
              'probs_squeeze_ratio', 'probs_squeeze_activation_cls', 'probs_output_activation_cls', 'left_mul',
              'dim_per_head_v', 'value_gate_activation_cls',
              'float32_logits', 'float32_probs', 'float32_value', 'qk_norm',
              'shared_qk_dim', 'shared_ov_dim', 'dim_per_shared_head', 'scale_shared_key', 'scale_init', 'scale_bias', 'rotate_shared_qk',
              ]:
    NAME = name.upper()
    if hasattr(cls, NAME):
      setattr(transformer_layer_p.tr_atten_tpl, name, getattr(cls, NAME))

  transformer_layer_p.tr_fflayer_tpl.has_bias = not cls.USE_GATED_ACTIVATION or cls.USE_BIAS  # XD add
  if cls.ACTIVATION_CLS == layers.GELU: transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True  # XD: add if
  if not cls.USE_GATED_ACTIVATION: transformer_layer_p.hidden_dims = cls.MODEL_DIMS * 4 # XD add

  for atten_p in (
      transformer_layer_p.tr_atten_tpl,
      transformer_layer_p.cross_atten_tpl,
  ):
    if atten_p is None:
      continue
    atten_wp = atten_p.weight_split_dims_mapping
    atten_wp.proj = ['data', 'mdl', None]

  if task_p.early_stopping_fn is None:
    task_p.early_stopping_fn = pax_fiddle.Config(EarlyStoppingFn)
    task_p.early_stopping_fn.target_log_pplx = cls.TARGET_LOG_PPLX

  return task_p


@experiment_registry.register
class C4SpmdAdam(TransformerLmSpmdAdam,
                 C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer."""
  # VOCAB_SIZE = 50320  # XD: GPT2Tokenizer.vocab_size = 50257
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.SiLU  # XD: GELU, SiLU
  USE_GATED_ACTIVATION = True  # XD: False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)
    return task_p


class C4SpmdGpt3AdamOrgHP(C4SpmdAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Training target
  TARGET_LOG_PPLX = 2.69 - 2.69  # XD

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamOrgHPBS1p5k1536Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 1536 replicas with 1536 global batch size."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 64, 24]
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1

@experiment_registry.register
class C4SpmdGpt3SmallRoPE(C4SpmdGpt3AdamOrgHP):  # XD
  r"""small GPT-3 config with RoPE.
  """
  MAX_SEQ_LEN = 2048 // 2  # XD
  NUM_LAYERS = 12
  MODEL_DIMS = 768
  ACTIVATION_CLS = layers.SiLU  # layers.SiLU/GELU  # XD
  USE_GATED_ACTIVATION = True  # XD
  HIDDEN_DIMS = MODEL_DIMS * 4 # 2048  # XD: MODEL_DIMS * 4
  NUM_HEADS = 12
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  USE_BIAS = False # XD add
  NORMALIZATION_CLS = normalizations.RmsNorm  # XD add RmsNorm
  LEARNING_RATE = 2e-4  # XD
  PERCORE_BATCH_SIZE = 4
  FPROP_DTYPE = jnp.bfloat16

  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [1, 4, 2]

  SEPARATE_EMBEDDING = True  # XD
  USE_ROTARY_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 2048  # XD add

  SUMMARY_INTERVAL_STEPS = 10  # XD
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 1

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    if self.USE_ROTARY_POSITION_EMB: task_p.model.lm_tpl.position_emb_tpl = None  # XD: add if
    return task_p

@experiment_registry.register
class C4SpmdGpt37BRoPE(C4SpmdGpt3SmallRoPE):  # XD
  MAX_SEQ_LEN = 2048 #// 2  # XD
  VOCAB_SIZE = 32000
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  NUM_GROUPS = -1  # XD
  # DIMS_PER_HEAD = 128
  COMBINE_QKV = True
  
  PERCORE_BATCH_SIZE = 8 * 2
  #ICI_MESH_SHAPE = [1, 8 // NUM_GROUPS, NUM_GROUPS]
  # ICI_MESH_SHAPE = [4, 1, 8]  # bs=2, 0.146, combine_qkv 0.1514  by xd
  # ICI_MESH_SHAPE = [1, 8, 4]  # bs=8, 0.044, combine_qkv 0.045  by xd
  #ICI_MESH_SHAPE = [1, 32, 1]  # bs=4, combine_qkv 0.0935  by xd
  ICI_MESH_SHAPE = [1, 32, 1]  # v5-64 bs=8 0.131 by xd
  CHECKPOINT_EVERY_N_STEPS=50

@experiment_registry.register
class C4SpmdGpt37BRoPEv4(C4SpmdGpt37BRoPE):
  PERCORE_BATCH_SIZE = 8
  # ICI_MESH_SHAPE = [1, 32, 1]  # bs=16*2*16*1, v4-64: 0.175 by lsp seqlen 1024
  #ICI_MESH_SHAPE = [1, 16, 1]  # bs=4*1*16*1, v4-16: seqlen 1024 0.388  seqlen 2048 0.23
  # ICI_MESH_SHAPE = [1, 8, 2]  # v4-32 bs=8 0.119  by xd
  ICI_MESH_SHAPE = [1, 16, 1]  # v4-32 bs=8 0.138 by lsp
  # ICI_MESH_SHAPE = [1, 16, 2]  # v4-64 bs=8 0.121 by xd

@experiment_registry.register
class C4SpmdGpt313BRoPE(C4SpmdGpt3SmallRoPE):  # XD
  VOCAB_SIZE = 32000  # XD
  NUM_LAYERS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 40
  # DIMS_PER_HEAD = 128
  COMBINE_QKV = False
  
  PERCORE_BATCH_SIZE = 6
  ICI_MESH_SHAPE = [1, 16, 4]  # bs=6*8*8, 0.032

@experiment_registry.register
class C4SpmdLlamaMedium(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  HIDDEN_DIMS = 2816  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64
  # NUM_GROUPS = 1  # XD
  COMBINE_QKV = False
  
  # LEARNING_RATE = 6e-5
  LR_COS_WARMUP = 256   # XD
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 50000

  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 32, 1]  # 0.549， combine_qkv 0.493???!!!, v5 0.436???
  # ICI_MESH_SHAPE = [32, 1, 1]

@experiment_registry.register
class C4SpmdLlamaXL(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 28
  MODEL_DIMS = 2048
  HIDDEN_DIMS = 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64
  # NUM_GROUPS = 1  # XD
  COMBINE_QKV = False
  
  # LEARNING_RATE = 6e-5
  LR_COS_WARMUP = 256   # XD
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 65536

  PERCORE_BATCH_SIZE = 16  # 0.168, v4 0.189!?
  ICI_MESH_SHAPE = [1, 64, 1]

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  DIM_PER_SHARED_HEAD = 16
  FLOAT32_LOGITS = True

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x64(C4SpmdLlamaMediumShareHeads):
  SHARED_QK_DIM = 1024  # 0.233, float32_logits 0.218, float32_logits v4 0.287 / 0.309 (fix probs fp32->fp16)
  SHARED_OV_DIM = 1024
  NUM_SHARED_HEADS = 16
  DIM_PER_SHARED_HEAD = 64

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16(C4SpmdLlamaMediumShareHeads):
  DIMS_PER_HEAD = 48
  SHARED_QK_DIM = 256  # v4 0.514
  SHARED_OV_DIM = 256
  NUM_SHARED_HEADS = 16

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16NoRot(C4SpmdLlamaMediumShareHeads16x16):
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x2(C4SpmdLlamaMediumShareHeads):
  DIMS_PER_HEAD = 48
  SHARED_QK_DIM = 256  # 0.321
  SHARED_OV_DIM = 256
  NUM_SHARED_HEADS = 128
  DIM_PER_SHARED_HEAD = 2

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x2NoRot(C4SpmdLlamaMediumShareHeads128x2):
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x64FP32value(C4SpmdLlamaMediumShareHeads16x64):
  FLOAT32_VALUE = True  # v4 0.282

@experiment_registry.register
class C4SpmdLlamaMediumShareQK16x64(C4SpmdLlamaMediumShareHeads16x64):
  SHARED_OV_DIM = 0  # 0.298

@experiment_registry.register
class C4SpmdLlamaMediumShareOV16x64(C4SpmdLlamaMediumShareHeads16x64):
  SHARED_QK_DIM = 0  # 0.296

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  # 0.464
  SHARED_OV_DIM = 128
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 1
  SHARED_QK_DIM = 256  # 0.394
  SHARED_OV_DIM = 256
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x13ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 1
  SHARED_QK_DIM = 208  # 0.424
  SHARED_OV_DIM = 208
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8x2ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 2
  SHARED_QK_DIM = 256  # 0.462
  SHARED_OV_DIM = 256
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  # 0.461
  SHARED_OV_DIM = 128
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot16x8ScaleInit05_15(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  ROTATE_SHARED_QK = True  # 0.45

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8ScaleInit00_10(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  SCALE_BIAS = 0.05

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x1ScaleInit05_15(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  DIM_PER_SHARED_HEAD = 1  # 0.464

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x1ScaleInit25_75(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  SCALE_INIT = WeightInit.Uniform(0.25)
  SCALE_BIAS = 0.5

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot(C4SpmdLlamaMediumShareHeads):
  ROTATE_SHARED_QK = True  # 0.470

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot128(C4SpmdLlamaMediumShareHeads128):
  ROTATE_SHARED_QK = True  # 0.455

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsScaleKRot128(C4SpmdLlamaMediumShareHeadsRot128):
  SCALE_SHARED_KEY = True  # 0.436

@experiment_registry.register
class C4SpmdLlamaMediumDimPerHead128(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 128   # 192 0.4, 128 0.47
  # DIMS_PER_HEAD = 160  # 0.419

@experiment_registry.register
class C4SpmdLlamaMediumHead16x128(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 128   # v4 0.515

@experiment_registry.register
class C4SpmdLlamaMediumShareQK(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 96  # 0.207
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareOV(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_OV_DIM = 96  # 0.204

@experiment_registry.register
class C4SpmdLlamaMediumShareQK128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  #
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareOV128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_OV_DIM = 128  # 0.489

@experiment_registry.register
class C4SpmdLlamaMediumv4(C4SpmdLlamaMedium):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.619

@experiment_registry.register
class C4SpmdLlamaMediumGA(C4SpmdLlamaMedium):
  DIM_PER_HEAD_V = 128  # 0.6
  VALUE_GATE_ACTIVATION_CLS = layers.SiLU
  HIDDEN_DIMS = 1408

@experiment_registry.register
class C4SpmdLlamaMediumGAv4(C4SpmdLlamaMediumGA):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.618

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8(C4SpmdLlamaMediumGA):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256  # 0.642, v5 0.544???

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8FP32logits(C4SpmdLlamaMediumGA256x8):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256
  FLOAT32_LOGITS = True

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8QKNorm(C4SpmdLlamaMediumGA256x8):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256
  QK_NORM = True

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8v4(C4SpmdLlamaMediumGA256x8):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.733

@experiment_registry.register
class C4SpmdLlamaMediumResTH(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.37, res 0.208/0.211 
  PROJECT_LOGITS = True
  PROJECT_PROBS = True

@experiment_registry.register
class C4SpmdLlamaXLHead16x128(C4SpmdLlamaXL):
  NUM_HEADS = 16  # 0.20
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlamaXLFP32logits(C4SpmdLlamaXL):
  FLOAT32_LOGITS = True  # 0.155

@experiment_registry.register
class C4SpmdLlamaXLResTH(C4SpmdLlamaXL):
  NUM_GROUPS = 1  #  v4 0.150
  PROJECT_LOGITS = True
  PROJECT_PROBS = True
  LOGITS_ABSORB_RESIDUAL = True
  PROBS_ABSORB_RESIDUAL = True

@experiment_registry.register
class C4SpmdLlamaXLHead16x128ResTH(C4SpmdLlamaXLResTH):
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128
  
@experiment_registry.register
class C4SpmdLlamaMediumResTHAbsorbRes(C4SpmdLlamaMediumResTH):
  ABSORB_RESIDUAL = True  # 0.355 v4 0.449, v5 0.298~0.376?! very unstable

@experiment_registry.register
class C4SpmdLlamaMediumResTHFP32logitsprobs(C4SpmdLlamaMediumResTH):
  FLOAT32_LOGITS = True
  FLOAT32_PROBS = True  # 0.152

@experiment_registry.register
class C4SpmdLlamaMediumResTHAbsorbResFP32logitsprobs(C4SpmdLlamaMediumResTHAbsorbRes):
  FLOAT32_LOGITS = True
  FLOAT32_PROBS = True  # 0.343

@experiment_registry.register
class C4SpmdLlamaMediumResTHLeftMul(C4SpmdLlamaMediumResTH):
  LEFT_MUL = True  # v4 0.336

@experiment_registry.register
class C4SpmdLlamaMediumResTHv4(C4SpmdLlamaMediumResTH):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.336

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs(C4SpmdLlamaMediumResTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUProbs(C4SpmdLlamaXLResTH):
  LOGITS_SQUEEZE_RATIO = 2   # v4 0.112 v4 absorbres 0.115
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  LOGITS_ABSORB_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaXLTHLogitsFFN2GELUResProbs(C4SpmdLlamaXLResTHLogitsFFN2GELUProbs):
  LOGITS_RESIDUAL = False   # v4 0.154

@experiment_registry.register
class C4SpmdLlamaXLHead16x128THLogitsFFN2GELUResProbs(C4SpmdLlamaXLTHLogitsFFN2GELUResProbs):
  NUM_HEADS = 16  # 0.189
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsBS1(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PERCORE_BATCH_SIZE = 1

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsReLU(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU  # 0.311

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsv4(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.318, transpose 0.323, global transpose 0.322

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogits(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.307
  PROJECT_LOGITS = True 

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsv4(C4SpmdLlamaMediumResTHLogits):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.409

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsGaussian05(C4SpmdLlamaMediumResTHLogits):
  SCALE_INIT = WeightInit.Gaussian(0.05)

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ProbsFFN2(C4SpmdLlamaMediumResTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  PROBS_SQUEEZE_RATIO = 2    # v4
  PROBS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHLogits(C4SpmdLlamaMediumResTHLogits):
  LOGITS_RESIDUAL = False  # 0.38

@experiment_registry.register
class C4SpmdLlamaMediumTH(C4SpmdLlamaMediumResTH):
  LOGITS_RESIDUAL = False
  PROBS_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaMediumTHGaussian25(C4SpmdLlamaMediumTH):
  SCALE_INIT = WeightInit.Gaussian(0.25) # 0.383

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFNProbs(C4SpmdLlamaMediumTH):
  LOGITS_SQUEEZE_RATIO = 2   # v4 0.47
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFNProbsFFN(C4SpmdLlamaMediumTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  PROBS_SQUEEZE_RATIO = 2    # v4 0.437
  PROBS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHHEAD64x16Gaussian125(C4SpmdLlamaMediumTH):
  NUM_HEADS = 64
  DIMS_PER_HEAD = 16
  SCALE_INIT = WeightInit.Gaussian(0.125)  # 0.21

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN1GELU(C4SpmdLlamaMediumResTHLogits):
  SQUEEZE_RATIO = 1  # 0.279, bias 0.271
  SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELU(C4SpmdLlamaMediumResTHLogits):
  LOGITS_SQUEEZE_RATIO = 2  # 0.293, bias 0.290
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ReLU(C4SpmdLlamaMediumResTHLogits):
  LOGITS_SQUEEZE_RATIO = 2  # 
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ReLUv4(C4SpmdLlamaMediumResTHLogitsFFN2ReLU):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.386

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2(C4SpmdLlamaMediumResTHLogits):
  SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFN2GELU(C4SpmdLlamaMediumResTHLogitsFFN2GELU):
  LOGITS_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN4GELU(C4SpmdLlamaMediumResTHLogitsFFN2GELU):
  SQUEEZE_RATIO = 4  # 

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbs(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.304
  PROJECT_PROBS = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2(C4SpmdLlamaMediumResTHProbs):
  PROBS_SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2GELU(C4SpmdLlamaMediumResTHProbsFFN2):
  SQUEEZE_ACTIVATION_CLS = layers.GELU  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2v4(C4SpmdLlamaMediumResTHProbs):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.395
  SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2GELUv4(C4SpmdLlamaMediumResTHProbsFFN2v4):
  SQUEEZE_ACTIVATION_CLS = layers.GELU  # 0.385

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2ReLUv4(C4SpmdLlamaMediumResTHProbsFFN2v4):
  SQUEEZE_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumResTHFP32logits(C4SpmdLlamaMediumResTH):
  FLOAT32_LOGITS = True  # 0.149

@experiment_registry.register
class C4SpmdLlamaMediumResTHGaussian05(C4SpmdLlamaMediumResTH):
  SCALE_INIT = WeightInit.Gaussian(0.05)

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4(C4SpmdLlamaMediumResTH):
  SQUEEZE_RATIO = 4  # res 0.173

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4GELU(C4SpmdLlamaMediumResTH):
  SQUEEZE_RATIO = 4  # res 0.173
  SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHx2(C4SpmdLlamaMediumResTH):
  NUM_HEADS = 16 * 2  # res 0.105, v5 0.195

@experiment_registry.register
class C4SpmdLlamaMediumResTHx4(C4SpmdLlamaMediumResTH):
  NUM_HEADS = 16 * 4  # res 0.059

@experiment_registry.register
class C4SpmdGpt3XLRoPE(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  HIDDEN_DIMS = 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  # DIMS_PER_HEAD = 128
  
  PERCORE_BATCH_SIZE = 4 * 4
  ICI_MESH_SHAPE = [1, 8, 4]

@experiment_registry.register
class C4SpmdPipelineAdam(TransformerLmSpmdPipelineAdam, C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer with pipeline."""
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  MICROBATCH_SIZE = 2
  ICI_MESH_SHAPE = [2, 1, 2, 2]
  NUM_STAGES = 2
  EMB_W_DATA_DIMS = ('replica', 'data')

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = (
        GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    )
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class C4SpmdPipelineGpt3AdamOrgHP(C4SpmdPipelineAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = False

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Training target
  TARGET_LOG_PPLX = 2.69

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


class C4SpmdPipelineGpt3AdamMLPerfHP(C4SpmdPipelineGpt3AdamOrgHP):
  r"""GPT-3 config for MLPerf reference."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200
  FPROP_DTYPE = jnp.float32
  SUMMARY_INTERVAL_STEPS = 1
  # subclass must set the eval and the checkpoint intervals
  EVAL_INTERVAL_STEPS = None
  CHECKPOINT_EVERY_N_STEPS = None
  CHECKPOINT_MAX_TO_KEEP = 100

  # Let set_adam_and_learning_rate_schedule calculate the following HPs
  # based on global batch size
  LEARNING_RATE = None
  LR_COS_WARMUP = None
  LR_COS_DECAY_START = None
  LR_COS_DECAY_END = None


@experiment_registry.register
class C4SpmdPipelineGpt3AdamOrgHPBS1p5k768Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Using the orininal HP set.
  """
  PERCORE_BATCH_SIZE = 2
  VOCAB_SIZE = 51200
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIAZE = 8
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 50
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Following MLPerf training benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 2
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 16
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS2k512Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 512 replicas with 2k global batch size.

  Following MLPerf training benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 8]
  # NUM_MICROBATCHS = 256
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 12
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS3k768Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 3072 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 16
  EVAL_INTERVAL_STEPS = 8
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 4096 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 6
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 8192 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 8
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 16
  EVAL_INTERVAL_STEPS = 3
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4Spmd1BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 1B params.

  Model Parameters:  Global batch size = 1 * 4 * 1 * 32 = 128
  """
  NUM_LAYERS = 13
  MODEL_DIMS = 2560
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 20
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  # VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]


@experiment_registry.register
class C4Spmd1BAdam4ReplicasLimitSteps(C4Spmd1BAdam4Replicas):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 15000
    return task_p


@experiment_registry.register
class C4Spmd2BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 2B params.

  Model Parameters: Global batch size = 1 * 4 * 1 * 32 = 128.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]

@experiment_registry.register
class C4Spmd2BAdam32Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 1 * 8 * 4 * 8 = 256.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024 * 2  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 8, 4]
  
@experiment_registry.register
class C4Spmd2BAdam32x2Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 1 * 16 * 2 * 16 = 512.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024 * 2  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]

@experiment_registry.register
class C4SpmdLLaMA7BAdam32Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 4 * 1 * 8 * 1 / 8 = 4.
  """
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 1  # 4
  MAX_SEQ_LEN = 2048  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [4, 1, 8]

  # def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  #   task_p = super().task()
  #   task_p.train.num_train_steps = 30
  #   return task_p

@experiment_registry.register
class C4SpmdLLaMA1BAdam32Replicas(C4SpmdLLaMA7BAdam32Replicas):  # XD
  r"""
  Model Parameters: Global batch size = 4 * 1 * 8 * 1 / 8 = 4.
  """
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4  # 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8  # 4
  COMBINE_QKV = False

  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [16, 1, 2]

@experiment_registry.register
class C4Spmd16BAdam32Replicas(C4SpmdAdam):
  r"""GPT-3 config with 16B params.

  Model Parameters: Global batch size = 1 * 2 * 16 * 16 = 512.
  """
  NUM_LAYERS = 36
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 48
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16 #// 8 # XD: v4->v3
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]


@experiment_registry.register
class C4Spmd32BAdam64Replicas(C4SpmdAdam):
  r"""GPT-3 config with 32B params.

  Model Parameters: Global batch size = 1 * 16 * 4 * 8 = 512.
  """
  NUM_LAYERS = 40
  MODEL_DIMS = 8192
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 64 replicas with 192 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 3
  EVAL_INTERVAL_STEPS = 25000
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdPipelineGpt3SmallAdam8Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """

  NUM_STAGES = 2
  NUM_LAYERS = 24
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 64
  MICROBATCH_SIZE = 8
  FPROP_DTYPE = jnp.bfloat16
  LEARNING_RATE = 2.0e-4
  ICI_MESH_SHAPE = [2, 1, 2, 2]

  CHECKPOINT_MAX_TO_KEEP = 1000
  EVAL_INTERVAL_STEPS = 10
  SUMMARY_INTERVAL_STEPS = 5
  CHECKPOINT_EVERY_N_STEPS = 200
