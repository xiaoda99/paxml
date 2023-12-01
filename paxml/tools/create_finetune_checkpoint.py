import json
import os
from collections import defaultdict
from smart_open import open
from flax.traverse_util import flatten_dict, unflatten_dict
import lingvo
import optax
import orbax.checkpoint
import orbax
import jax
from copy import copy, deepcopy


from collections import defaultdict
import numpy as np
import re
import contextlib
import typing
from typing import Type

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
from paxml import base_experiment
from paxml import checkpoint_creators
from paxml import checkpoint_types
from paxml import decode_programs as decode_programs_lib
from paxml import executors
from paxml import experiment_utils
from paxml import partitioning
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import pax_fiddle
from praxis import pytypes
from praxis import py_utils
import tensorflow.compat.v2 as tf

from paxml import checkpoints  # mapped to internal
from datetime import datetime

Dict2NestedMap = lingvo.core.nested_map._FromNestedDict

JTensor = pytypes.JTensor
Checkpointer = checkpoints.Checkpointer
CheckpointType = checkpoints.CheckpointType
instantiate = base_hyperparams.instantiate
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler

RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
# pylint: disable=protected-access
_checkpoint_dir = checkpoint_creators._checkpoint_dir
_create_checkpointer = checkpoint_creators._create_checkpointer

def load_pretrained_model(model_path):
#     model_path = 'gs://llm_projects_us-central2/log/C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormv4_nocap/checkpoints/checkpoint_00066300'
    read_dir = '/'.join(model_path.split('/')[:-1])
    model_name = model_path.split('/')[-3]#read_dir.split('/')[-2]
    step_prefix = "checkpoint"
    step_format_fixed_length = 8
    load_step = int(model_path.split('/')[-1].split('_')[-1])

    options = orbax.checkpoint.CheckpointManagerOptions(
        step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length
    )
    item = {
        "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    }
    mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)
    
    checkpoint_name = f"{step_prefix}_" + str(load_step).zfill(step_format_fixed_length)
    
    print(f"checkpoint_name: {checkpoint_name}")
    metadata_path = os.path.join(read_dir, checkpoint_name, "metadata/metadata")
    print(f"metadata_path: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print('metadata', metadata)
    
    flat_metadata = flatten_dict(metadata["train_state_metadata"])
    unpadded_global_shapes = defaultdict(dict)
    for k, v in flat_metadata.items():
        param_key, shape_dtype = k[:-1], k[-1]
        if shape_dtype in ["unpadded_shape", "dtype"]:
            unpadded_global_shapes[param_key][shape_dtype] = v
        shape_dtype = unpadded_global_shapes[param_key]
        if len(shape_dtype) == 2:
            shape_dtype = jax.ShapeDtypeStruct(
                shape=shape_dtype["unpadded_shape"], dtype=shape_dtype["dtype"]
            )
            unpadded_global_shapes.update({param_key: shape_dtype})
        
    # load model
    unflat_unpadded_global_shapes = unflatten_dict(unpadded_global_shapes)
    with jax.default_device(jax.devices("cpu")[0]):
        weights = mngr.restore(load_step, items={"state": unflat_unpadded_global_shapes})
    return weights


def load_exp_state(exp, step):
    _exp, tpu_type = get_tpu_type(exp)
    experiment_config = get_experiment(f'paxml.tasks.lm.params.c4.{_exp}')()
    suffix = f'_{global_cfg.tputype2zone[tpu_type]}' if tpu_type in ['v4', 'v5'] else ''  # append_zone
    job_log_dir = epath.Path(f'gs://llm_projects{suffix}/log/{exp}/')
    experiment_config.ICI_MESH_SHAPE = [1, 8, 1]
    experiment_config.PERCORE_BATCH_SIZE = 1 #/ 8 # 16
    task_p = experiment_config.task()
    task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)
    jax_task = instantiate(task_p)
    input_p = experiment_config.datasets()
    train_input_p = [v for v in input_p if not v.is_training][0]
    maybe_use_persistence_checkpointing = False
    checkpoint_type = checkpoint_types.retrieve_checkpoint_type(maybe_use_persistence_checkpointing, jax_task)
    checkpointer = _create_checkpointer(task_p, job_log_dir, checkpoint_type, None, train_input_p=train_input_p,
        enable_async_checkpointing=False,enable_checkpoint_saving=True, enforce_restore_shape_check=False,
        maybe_use_persistence_checkpointing=maybe_use_persistence_checkpointing,tensorstore_use_ocdbt=False,
    )
    partitioner = partitioning.create_partitioner(jax_task, reshard_inputs=True, auto_sharding_mode=None)
    train_program = experiment_config.train_program()
    executor = executors.DefaultExecutor()

    self = executor
    self._task = jax_task
    self._job_log_dir = job_log_dir
    self._partitioner = partitioner
    self._train_program = train_program

    root_prng_key = jax.random.PRNGKey(self._task.train.random_seed)
    train_input_p = partitioner.preprocess_input_config(train_input_p)
    train_input_p.num_batches_to_skip = 0
    train_input, train_input_for_partitioner, train_input_for_checkpoint = \
    self._maybe_create_train_input(self._task, checkpointer.step_to_restore, train_input_p)
    train_input_specs = None
    partitioner.setup(jax_task, root_prng_key,
        train_inputs_shape_dtype=train_input_specs,
        train_input_pipeline=train_input_for_partitioner,
        job_log_dir=job_log_dir,
    )
    train_state_metadata = partitioner.get_train_state_metadata(discard_opt_states=False)
    if step ==0:
        checkpointer._step_to_restore = None
        partitioned_train_state, train_state_provenance, total_num_params, root_prng_key = checkpointer.get_model_states(
        partitioner, train_state_metadata, root_prng_key, train_input_for_checkpoint)
    else:
        checkpointer._step_to_restore = step
        partitioned_train_state, train_state_provenance, total_num_params, root_prng_key = checkpointer.get_model_states(
            partitioner, train_state_metadata, root_prng_key, train_input_for_checkpoint)
        print(root_prng_key)
    return partitioned_train_state, train_state_metadata, checkpointer, partitioner, train_input_for_checkpoint, task_p


def fuse_mdl_state(weight, pretrained_weight):
    flat_pretrained_weight = flatten_dict(pretrained_weight)
    flat_weight = flatten_dict(weight)
    for k,v in flat_pretrained_weight.items():
        _k = k[2:]
        assert _k in flat_weight
        flat_weight[_k] = v
    return unflatten_dict(flat_weight)

def fuse_all_states(partitioned_train_state_f, partitioned_train_state_l, multi_opt=False, fuse_opt=True):
    #fuse mdl states
    mdl_states_f = flatten_dict(partitioned_train_state_f.mdl_vars)
    mdl_states_l = flatten_dict(partitioned_train_state_l.mdl_vars)
    mdl_states_f.update(mdl_states_l)
    mdl_states_f = unflatten_dict(mdl_states_f)
    #fuse opt states
    if fuse_opt:
        opt_states_f = deepcopy(partitioned_train_state_f.opt_states[0]) # dict type if not multi_opt else MaskedState type
        opt_states_l = partitioned_train_state_l.opt_states[0]
        for k in opt_states_l.keys(): # 'no_prefix', 'p#28#i-1'
            print(k, len(opt_states_l[k]))
            if multi_opt:
                opt_states_f[k] = list(opt_states_f[k])
                for j in range(len(opt_states_f[k])):
                    inner_state = list(opt_states_f[k][j].inner_state)
                    for i in [0,1,3]: # update count
                        inner_state[i] = Dict2NestedMap(opt_states_l[k][i])
                    if j==0: # main_optimizer
                        flat_l = flatten_dict(opt_states_l[k][2])
                        flat_f = flatten_dict(inner_state[2])
                        flat_f.update(flat_l)
                        for _k,_v in flat_f.items():
                            if isinstance(_v, optax.MaskedNode):
                                flat_f[_k] = 'MaskedNode'
                        unflat_f = Dict2NestedMap(unflatten_dict(flat_f))
                        inner_state[2] = unflat_f.Transform(lambda x: optax.MaskedNode() if isinstance(x, str) and x == 'MaskedNode' else x)
                    # update inner opt state
                    opt_states_f[k][j] = opt_states_f[k][j]._replace(inner_state=tuple(inner_state))
                opt_states_f[k] = tuple(opt_states_f[k])
            else:
                opt_states_f[k] = list(opt_states_f[k])
                for i in [0,1,3]: # update count
                    opt_states_f[k][i] = opt_states_l[k][i]
                flat_f = flatten_dict(opt_states_f[k][2])
                flat_l = flatten_dict(opt_states_l[k][2])
                flat_f.update(flat_l)
                opt_states_f[k][2] = unflatten_dict(flat_f)
                opt_states_f[k] = tuple(opt_states_f[k])
        if multi_opt: 
            nm = lingvo.core.nested_map.NestedMap()
            for k, v in opt_states_f.items():
                nm[k] = v
            opt_states_f = nm
        partitioned_train_state_f2 = partitioned_train_state_f.replace(step=partitioned_train_state_l.step, mdl_vars=mdl_states_f,opt_states=[opt_states_f])
    else:
        partitioned_train_state_f2 = partitioned_train_state_f.replace(step=partitioned_train_state_l.step, mdl_vars=mdl_states_f)
    return partitioned_train_state_f2

def test_regexp(partitioned_train_state, regexp='.*(pre|post)_proj.*'):
    prefix = py_utils.extract_prefixed_keys_from_nested_map(partitioned_train_state.mdl_vars)
    regexp = re.compile(regexp)
    mask = jax.tree_map(
              lambda x, regexp=regexp: regexp.match(x) is not None, prefix
          )
    return mask

def main():
    exp_llama = 'C4SpmdLlamaXLHead16x128'; step_llama =70600
    # exp_finetune = 'C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormFinetune'; 
    # exp_finetune = 'C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormFinetuneDDHid'; 
    exp_finetune = 'C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormFinetuneNoresLowlr'; 
    step_finetune = 0

    # load pretrained model state
    partitioned_train_state_l, train_state_metadata_l, checkpointer_l, partitioner_l, train_input_for_checkpoint, task_p_l = load_exp_state(exp_llama, step_llama)
    # init model state to be finetuned
    partitioned_train_state_f,  train_state_metadata_f,  checkpointer_f, partitioner_f, train_input_for_checkpoint_f, task_p_f = load_exp_state(exp_finetune, step_finetune)

    # transfer pretrained state to initialized finetue state
    partitioned_train_state_f = fuse_all_states(partitioned_train_state_f, partitioned_train_state_l, multi_opt=False, fuse_opt=False)

    # check
    print('model parapmeters:')
    for k,v in flatten_dict(partitioned_train_state_f.mdl_vars).items():
        print('.'.join(k), v.shape, v.std())
    print('opt state keys': partitioned_train_state_f.opt_states[0].keys())
    print('opt state:')
    for k,v in flatten_dict(partitioned_train_state_f.opt_states[0]['p#28#i-1'][2]).items():
        print('.'.join(k), (v.shape, v.std()) if hasattr(v, 'shape') else v) 

    # save merged finetune state
    train_input_pipeline = None
    print('save directory:', checkpointer_f.checkpoint_manager.directory)
    print('save step:', partitioned_train_state_f.step)
    print('saving')
    checkpointer_f.checkpoint_manager.save(partitioned_train_state_f.step, partitioned_train_state_f, train_state_metadata_f.unpadded_global_shapes, train_input_pipeline)
    print('saving done')

if __name__ == '__main__':
    main()
