import numpy as np

import jax
# jax.config.update('jax_array', False)
from jax import pmap
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from jax.experimental import PartitionSpec as P
from jax.experimental import multihost_utils
from jax.experimental.multihost_utils import host_local_array_to_global_array, global_array_to_host_local_array

print(f'n_devices = {len(jax.devices())}, pid {jax.process_index()}/{jax.process_count()}')
# mesh_shape = (16, 2)
# devices = np.asarray(jax.devices()).reshape(*mesh_shape)
# mesh = Mesh(devices, ('x', 'y'))

# with mesh:
#     rank = jax.process_index(); print(rank)
#     x = np.arange(rank * 16, (rank + 1) * 16).reshape(8, 2)#; x
#     # f = pjit(lambda x: x, in_axis_resources=P(('x', 'y'),), out_axis_resources=P('x', 'y'))
#     def f(x): return x #.mean()
#     f_pjit = pjit(f, in_axis_resources=P('x'), out_axis_resources=P('x'))
#     x = host_local_array_to_global_array(x, mesh, P('x'))
#     y = f_pjit(x)
#     y = global_array_to_host_local_array(y, mesh, P('x'))
#     print('y =', y)
