# coding=utf-8
# Copyright 2024 The XXXX-1 Research Authors.
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

"""Run feature selection with Sequential Attention."""

import os

from absl import app

from sand.experiments import hyperparams


def get_cmd(params):
  """Get command for running experiments with input parameters."""
  experiments_dir = os.path.dirname(os.path.realpath(__file__))
  experiment_file = os.path.join(experiments_dir, 'run.py')
  cmd = ['python', experiment_file]
  for param in params:
    cmd.append(f'--{param}={params[param]}')
  return ' '.join(cmd)


def main(_):
  base_name = 'SAND'
  base_dir = './temp'
  parameters = []
  def get_params(k, name, seed):
    return {
        'data_name': name,
        'algo': 'sand',  # sa, lly, seql, gl
        'deep_layers': hyperparams.DEEP_LAYERS[name],
        'batch_size': hyperparams.BATCH[name],
        'num_epochs_select': hyperparams.EPOCHS[name],
        'num_epochs_fit': hyperparams.EPOCHS_FIT[name], # not used by sand
        'learning_rate': hyperparams.LEARNING_RATE[name],
        'alpha': 0,
        'decay_rate': 1,
        'decay_steps': 10000,
        'num_selected_features': k,
        'seed': seed,
        'enable_batch_norm': False,
        'num_inputs_to_select_per_step': 1,
        'model_dir': f'{base_dir}/{name}/{k}/{base_name}_seed_{seed}/',
        'sigma': hyperparams.SIGMA[name], # only used by sand
    }

  for seed in range(1,11):
    for k in [60]:
      parameters += [
          get_params(k, name, seed)
          for name in ['mice', 'mnist', 'fashion', 'isolet', 'coil', 'activity']
      ]

  for params in parameters:
    cmd = get_cmd(params)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
  app.run(main)
