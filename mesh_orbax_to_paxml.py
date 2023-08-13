import sys
import time

import jax
import numpy as np
import jax.numpy as jnp
# import orbax
# from optax import MaskedNode
from etils import epath

from praxis import base_hyperparams
# from praxis import pax_fiddle
from praxis import py_utils
from paxml import checkpoints  # mapped to internal
from paxml import checkpoint_managers
from paxml import train_states
from paxml import trainer_lib
# from flax.traverse_util import flatten_dict, unflatten_dict

sys.path.append('/home/lishengping/projects/paxml/paxml')

from paxml.main import get_experiment


try:
    jax.distributed.initialize()
except Exception as error:
    print(f'Error: {error}')
    assert jax.local_device_count() == 8
    

TrainState = train_states.TrainState
instantiate = base_hyperparams.instantiate
CheckpointType = checkpoints.CheckpointType
Checkpointer = checkpoints.Checkpointer
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
NestedMap = py_utils.NestedMap
checkpoint_type = CheckpointType.GDA
SAVE_INTERVAL_STEPS = 1

exp = 'C4SpmdLlamaMediumResTHv4'
experiment_config = get_experiment(f'tasks.lm.params.c4.{exp}')()
task_p = experiment_config.task()
jax_task = instantiate(task_p)

options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=10,
      save_interval_steps=SAVE_INTERVAL_STEPS,
      cleanup_tmp_directories=True,
  )
checkpointer = Checkpointer(
          PaxCheckpointHandler(
              enforce_restore_shape_check=False,
              use_ocdbt=False,
          )
      )
job_log_dir = epath.Path(f'gs://llm_projects/log/{exp}_skip/checkpoints')
step = 44000
# job_log_dir = epath.Path('gs://llm_base_models/baichuan-7B-easylm')
checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
      job_log_dir,
      checkpointer,
      train_input_checkpointer=False,
      options=options,
      checkpoint_type=checkpoint_type,
      tensorstore_use_ocdbt=False,
  )

start = time.time()
print(f'Start load pretrained model params....')
gold_mngr_dir = epath.Path('gs://llm_base_models/baichuan-7B-easylm')
gold_mngr_dir = epath.Path('gs://llm_base_models/orbax_async_test')
gold_item = {
            'params': orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
                }
gold_mngr = orbax.checkpoint.CheckpointManager(gold_mngr_dir, gold_item)

with jax.default_device(jax.devices("cpu")[0]):
    gold_w = gold_mngr.restore(gold_mngr.latest_step())
print(f'Load pretrained model params finished, take time: {time.time() - start}s.')

paxml_to_mesh_format = {
        ('params', 'lm', 'embedding_lookup', 'emb_var'): 'wte',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'ff_layer', 'ffn_layer1', 'linear', 'w'): 'w3',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'ff_layer', 'ffn_layer1_gate', 'linear', 'w'): 'w1',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'ff_layer', 'ffn_layer2', 'linear', 'w'): 'w2',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'self_attention', 'query', 'w'): 'wq',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'self_attention', 'key', 'w'): 'wk',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'self_attention', 'value', 'w'): 'wv',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'self_attention', 'post', 'w'): 'wo',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'layer_norm', 'scale'): 'attention_norm',
        ('params', 'lm', 'transformer', 'repeat', 'sub', 'x_layers_0', 'ff_layer', 'layer_norm', 'scale'): 'ffn_norm',
        ('params', 'lm', 'final_ln', 'scale'): 'ln_f',
        ('params', 'lm', 'softmax', 'logits_ffn', 'linear', 'w'): 'lm_head',
    }

num_heads = experiment_config.NUM_HEADS
model_dims = experiment_config.MODEL_DIMS 
head_dim = model_dims // num_heads

trans_result = {}
with jax.default_device(jax.devices("cpu")[0]):
    for k, v in paxml_to_mesh_format.items():
        values = []
        for gold_key, glod_values in flatten_dict(gold_w['params']).items():
            if v in gold_key:
                if v in 'wqwkwvwo':
                    glod_values = glod_values.reshape(model_dims, num_heads, head_dim)
                values.append([gold_key, glod_values])
        values = sorted(values, key=lambda x: x[0])
        if len(values) > 1:
            stack_values = np.stack(list(zip(*values))[1])
        else:
            stack_values = values[0][1]
        trans_result[k] = stack_values
    opt_state_mv = jax.tree_map(lambda x: jnp.zeros_like(x), trans_result)

print(f'Please simple check model shape and dtype...')
for k, v in trans_result.items():
    print(k, v.shape, v.dtype)

latest_step =  checkpoint_manager.latest_step()
step = latest_step + SAVE_INTERVAL_STEPS if latest_step is not None else SAVE_INTERVAL_STEPS
print(f'Model save step is {step}')
n_layers = experiment_config.NUM_LAYERS # 模型的层数
# n_layers = 32 # 模型的层数
check_saved_model_fail_or_success = True
start = time.time()
temp_no_prefix, temp_other = {}, {}
for key_tuple, param in opt_state_mv.items():
    if 'repeat' in key_tuple:
        temp_no_prefix[key_tuple] = MaskedNode()
        temp_other[key_tuple] = param
    else:
        temp_no_prefix[key_tuple] = param
        temp_other[key_tuple] = MaskedNode()

temp_no_prefix = unflatten_dict(temp_no_prefix)
temp_other = unflatten_dict(temp_other)
    
no_prefix = {'count': jnp.array(step), 'm': temp_no_prefix, 'v': temp_no_prefix}
other = {'count': jnp.array([step] * n_layers), 'm': temp_other, 'v': temp_other}
trans_opt_states = {
    'no_prefix': [{'count': jnp.array(step)}] * 2 + [no_prefix, {'count': jnp.array(step)}], 
    f'p#{n_layers}#i-1': [{'count': jnp.array([step] * n_layers)}] * 2 + [other, {'count': jnp.array([step] * n_layers)}], 
}
trans_opt_states = [trans_opt_states]
new_trainstate = TrainState(
                            step=jnp.array(step), 
                            mdl_vars=unflatten_dict(trans_result),
                            opt_states=trans_opt_states
)
padded_global_shapes = jax.tree_map(lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype) 
                                    if hasattr(x, 'shape') else x , new_trainstate)
checkpoint_manager.save(step, new_trainstate, padded_global_shapes, train_input_pipeline=None, force=False)
print(f'Saved model finished. take time: {time.time() - start}s...')

if check_saved_model_fail_or_success:
    start = time.time()
    print(f'Args check_saved_model_fail_or_success is {check_saved_model_fail_or_success}, start to check model whether saved successful...')
    # fake输入只是为了拿到dtype和shape
    seed = 0
    jax.random.PRNGKey(seed)
    low, high = 0, experiment_config.VOCAB_SIZE
    seq_length = 10
    # batch_size = experiment_config.PERCORE_BATCH_SIZE * 8
    batch_size = 1
    fake_input = {}
    fake_input['ids'] = np.random.randint(low, high, (batch_size, seq_length)).astype(np.int32)
    fake_input['labels'] = fake_input['ids'].astype(np.int32)
    fake_input['weights'] = np.ones((batch_size, seq_length)).astype(np.float32)
    fake_input['paddings'] = fake_input['weights']
    fake_input['segment_ids'] = fake_input['weights'].astype(np.int32)
    fake_input['segment_pos'] = np.arange(seq_length).reshape(1, -1).repeat(batch_size, axis=0).astype(np.int32)
    fake_input['_seqio_provenance/shard_index'] = np.array([-1]).repeat(batch_size).astype(np.int32)
    fake_input['_seqio_provenance/num_shards'] = fake_input['_seqio_provenance/shard_index']
    fake_input['_seqio_provenance/index_within_shard'] = fake_input['_seqio_provenance/shard_index'].astype(np.int64)
    fake_input['eval_sample_weights'] = fake_input['_seqio_provenance/shard_index'].astype(np.float32)
    fake_input = NestedMap(fake_input)
    
    inputs_shape_dtype = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype),
            fake_input,
        )
    train_state_metadata = trainer_lib.create_train_state_metadata(
        jax_task,
        inputs_shape_dtype,
        discard_opt_states=False,
        do_eval=True,
    )
    print(f'Start load model to check whether saved model is True or False...')
    device_mesh = py_utils.create_device_mesh(
          jax_task.model.ici_mesh_shape,
          jax_task.model.dcn_mesh_shape,
          contiguous_submeshes=jax_task.model.contiguous_submeshes,
      )
    global_mesh = jax.sharding.Mesh(device_mesh, jax_task.model.mesh_axis_names)
    restore_kwargs = {
              'version': 1.1,
            #   'specs': train_state_metadata.partition_specs, # shard
            #   'mesh': global_mesh, # mesh
              'transforms': None, # None
          }
    restore_kwargs = {'state': restore_kwargs}
    items = {'state': train_state_metadata.padded_global_shapes}
    restored_model = checkpoint_manager._manager.restore(step, items=items, restore_kwargs=restore_kwargs)
    print(f'Check model finished. model is  saved successfully. take time: {time.time() - start}s...')



jax.tree_util.tree_map(lambda x: x.shape, restored_model['state'].mdl_vars['params']['lm'])
jax.tree_util.tree_map(lambda x: x.shape, restored_model['state'].mdl_vars['params']['lm'])['transformer']['repeat']['sub']['x_layers_0']['self_attention']['pre_proj']
restored_model['state'].mdl_vars['params']['lm']['transformer']['repeat']['sub']['x_layers_0']['self_attention']['pre_proj']['w']

from tensorflow.python.lib.io import file_io
pre_proj = restored_model['state'].mdl_vars['params']['lm']['transformer']['repeat']['sub']['x_layers_0']['self_attention']['pre_proj']
np.save(file_io.FileIO(job_log_dir / 'pre_proj.npy', 'w'), pre_proj['w'])

post_proj = restored_model['state'].mdl_vars['params']['lm']['transformer']['repeat']['sub']['x_layers_0']['self_attention']['post_proj']
np.save(file_io.FileIO(job_log_dir / 'post_proj.npy', 'w'), post_proj['w'])
