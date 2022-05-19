'''
Test
'''
import argparse

from itertools import product
from p_tqdm import p_map
from pathlib import Path
from subprocess import Popen, DEVNULL

MODELS = {'sicyclegan': 'self_inverse_cycle_gan', 'cyclegan': 'cycle_gan',
          'cut': 'cut'}

def _gen_output(**kwargs):
  Popen(('python test.py ' +
         ' '.join(f'--{k} {v}' for k, v in kwargs.items())).split(),
        stdout=DEVNULL).wait()

def _get_model(name):
  for model_name, model in MODELS.items():
    if model_name in name:
      return model
  raise KeyError(f'Invalid experiment name: {name}')

def gen_output(kwargs):
  kwargs['model'] = _get_model(kwargs['name'])

  _gen_output(**kwargs)

def main(args):
  dataroot = args.dataroot
  dataset_mode = args.dataset_mode
  checkpoints_dir = args.checkpoints_dir
  output_dir = args.output_dir
  direction = args.direction
  experiments = args.experiments
  epochs = args.epochs
  phases = args.phases
  gpu_ids = args.gpu_ids
  num_test = args.num_test
  workers = args.workers

  def to_para(args):
    expr, epoch, phase = args
    return {
      'dataroot': dataroot,
      'name': expr,
      'checkpoints_dir': checkpoints_dir,
      'phase': phase,
      'dataset_mode': dataset_mode,
      'direction': direction,
      'results_dir': output_dir,
      'num_test': num_test,
      'epoch': epoch,
      'gpu_ids': gpu_ids,
    }

  p_map(gen_output, list(map(to_para, product(experiments, epochs, phases))),
        num_cpus=workers)

def get_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('experiments', type=str, nargs='+')
  parser.add_argument('--epochs', required=True, type=int, nargs='+')
  parser.add_argument('--phases', required=True, type=str, nargs='+',
                      choices=('train', 'test'))
  parser.add_argument('--num_test', type=int, default=10)
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--gpu_ids', type=str, default='-1')
  parser.add_argument('--dataroot', required=True, type=Path)
  parser.add_argument('--dataset_mode', required=True, type=str)
  parser.add_argument('--output-dir', required=True, type=Path)
  parser.add_argument('--direction', required=True, type=str, default='AtoB',
                      choices=('AtoB', 'BtoA'))
  parser.add_argument('--checkpoints-dir', required=True, type=Path,
                      default='checkpoints/')

  return parser

if __name__ == '__main__':
  main(get_parser().parse_args())
