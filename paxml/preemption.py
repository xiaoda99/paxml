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

"""Utils for preemption."""
import jax
from jax.experimental import multihost_utils


def reached_preemption_sync_point(step: int) -> bool:
  """Determine whether all hosts have reached a preemption sync point."""
  return (getattr(jax.config, 'jax_coordination_service', True) and
          multihost_utils.reached_preemption_sync_point(step))
