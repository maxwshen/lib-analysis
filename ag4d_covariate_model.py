# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
from collections import defaultdict
from mylib import util
import pandas as pd

import autograd
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Default params
inp_dir = _config.OUT_PLACE + 'ag4c_covariate_form_data/'
NAME = util.get_fn(__file__)
out_place = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_place)

import _data

# Global variables
log_fn = None
rs = None
editors, batches, names = None, None, None

def nonneg_transform(x):
  return x**2
  # return 0.5 * (np.tanh(x) + 1.0)

################################################################
# Indexing: Positions, nucleotides, etc
################################################################
lib_ranges = {
  'LibA': list(range(-10, 44 + 1)),
  '12kChar': list(range(-21, 32 + 1)),
  'LibA': list(range(-10, 45 + 1)),
  'LibA': list(range(-10, 45 + 1)),
}
leftmost_pos = min([min(lib_ranges[lib]) for lib in lib_ranges])
rightmost_pos = max([max(lib_ranges[lib]) for lib in lib_ranges])
num_nts_in_range = rightmost_pos - leftmost_pos + 1

def pos_to_idx(pos):
  return pos - leftmost_pos

def idx_to_pos(idx):
  return idx + leftmost_pos

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

sub_muts = [
  'AC', 'AG', 'AT', 
  'CA', 'CG', 'CT',
  'GA', 'GC', 'GT',
  'TA', 'TC', 'TG',
]
def mut_to_idx(ref_nt, obs_nt):
  key = '%s%s' % (ref_nt, obs_nt)
  return sub_muts.index(key)

def idx_to_mut(idx):
  return sub_muts[idx]

def editor_to_idx(editor):
  return editors.index(editor)

def idx_to_editor(idx):
  return editors[idx]

def batch_to_idx(batch):
  return batches.index(batch)

def idx_to_batch(idx):
  return batches[idx]

def name_to_idx(name):
  return names.index(name)

def idx_to_name(idx):
  return names[idx]


################################################################
# Setup environment
################################################################
def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))

def copy_script(out_dir):
  src_dir = _config.SRC_DIR
  script_nm = __file__
  subprocess.call('cp ' + src_dir + script_nm + ' ' + out_dir, shell = True)
  return

def print_and_log(text):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return

def setup_outdir():
  num_folds = count_num_folders(out_place)
  out_letters = alphabetize(num_folds + 1)

  out_dir = out_place + out_letters + '/'
  util.ensure_dir_exists(out_dir)
  copy_script(out_dir)
  
  out_dir_params = out_place + out_letters + '/parameters/'
  util.ensure_dir_exists(out_dir_params)

  global log_fn
  log_fn = out_dir + '_log_%s.out' % (out_letters)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('out dir: ' + out_letters)

  return out_dir, out_letters, out_dir_params

################################################################
# Data manipulation
################################################################
def load_data():
  covariates = pd.read_csv(inp_dir + 'X.csv', index_col = 0)
  with open(inp_dir + 'Y.pkl', 'rb') as f:
    Y_human = pickle.load(f)
  return covariates, Y_human

def transform_to_machine_data(covariates, Y_human):
  full_Y = pd.DataFrame()
  for nm in Y_human:
    df = Y_human[nm]
    df['Name'] = nm

    x_row = covariates[covariates['Name'] == nm]
    df['Batch'] = x_row['Batch'].iloc[0]
    df['Editor'] = x_row['Editor'].iloc[0]

    full_Y = full_Y.append(df, ignore_index = True)

  full_Y = full_Y.dropna()
  full_Y = full_Y[~full_Y['Position'].isin([22, 23])]
  Y = np.array(full_Y['Mean activity'])
  guide = full_Y
  print('Y shape: ', Y.shape)
  return Y, guide

def init_params(guide):
  global editors, batches, names
  editors = sorted(set(guide['Editor']))
  batches = sorted(set(guide['Batch']))
  names = sorted(set(guide['Name']))
  print('Num. editors: %s' % (len(editors)))
  print('\t' + '\n\t'.join(editors))
  print('Num. batches: %s' % (len(batches)))
  print('\t' + '\n\t'.join(batches))
  print('Num. experiment names: %s' % (len(names)))

  num_substitution_muts = 4 * 3
  editor_params = np.random.random(size = (len(editors), num_nts_in_range, num_substitution_muts))
  batch_params = np.random.random(size = (len(batches), num_nts_in_range, num_substitution_muts))
  activity_params = np.random.random(size = (len(names), ))
  background_param = np.random.random(size = (1, ))

  params = [
    editor_params,
    batch_params,
    activity_params,
    background_param,
  ]

  return params

def params_transform_allpos_allsubmut(param_subset, idx_to_nm_func):
  all_dfs = dict()
  for idx, sub_ps in enumerate(param_subset):
    editor_nm = idx_to_nm_func(idx)
    dd = defaultdict(list)
    for jdx in range(len(sub_ps)):
      pos = idx_to_pos(jdx)
      for kdx in range(len(sub_ps[jdx])):
        val = nonneg_transform(sub_ps[jdx][kdx])
        [ref_nt, obs_nt] = list(idx_to_mut(kdx))
        dd['Position'].append(pos)
        dd['Ref nt'].append(ref_nt)
        dd['Obs nt'].append(obs_nt)
        dd['Value'].append(val)
    df = pd.DataFrame(dd)
    all_dfs[editor_nm] = df
  return all_dfs


def params_transform_to_human(params):
  data = dict()
  '''
    Structure:
      Level 1: Parameter types, in ['Editor', 'Batch', 'Activity', 'Background']

    Level 2:
    data['Editor'][editor_nm] = DataFrame, columns = Position, Obs nt, Ref nt, Mean activity

    data['Batch'][batch_nm] = DataFrame, columns = Position, Obs nt, Ref nt, Mean activity

    data['Activity'][cond_name] = DataFrame, columns = Name, Activity

    data['Background'] = float, Background activity  

  '''
  [editor_params, batch_params, activity_params, background_param] = params
  data['Background'] = nonneg_transform(background_param)

  data['Editor'] = params_transform_allpos_allsubmut(editor_params, idx_to_editor)
  data['Batch'] = params_transform_allpos_allsubmut(batch_params, idx_to_batch)

  dd = defaultdict(list)
  for idx in range(len(activity_params)):
    name = idx_to_name(idx)
    val = nonneg_transform(activity_params[idx])
    dd['Name'].append(name)
    dd['Value'].append(val)
  data['Activity'] = pd.DataFrame(dd)

  return data

################################################################
# ADAM optimizer
################################################################
def exponential_decay(step_size):
  if step_size > 0.001:
      step_size *= 0.9995
  return step_size

def adam(grad, init_params_nn, callback = None, num_iters = 100, step_size = 0.001, b1 = 0.9, b2 = 0.999, eps = 10**-8):
  x_nn, unflatten_nn = flatten(init_params_nn)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  for i in range(num_iters):
    g_nn_uf = grad(unflatten_nn(x_nn), i)
    g_nn, _ = flatten(g_nn_uf)

    if callback: 
      callback(unflatten_nn(x_nn), i)
    
    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn      + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn**2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn = v_nn / (1 - b2**(i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)
  
  return unflatten_nn(x_nn)


################################################################
# Master
################################################################
def main_objective(params, Y, guide):
  num_samples = len(Y)
  total_loss = 0
  for y, (idx, guide_row) in zip(Y, guide.iterrows()):
    y_estimate = single_estimate(params, y, guide_row)
    total_loss += (y - y_estimate)**2
  final_loss = total_loss / num_samples
  # print(final_loss)
  # l2_reg_loss = L2_regularization(params) / num_samples
  # l2_lambda = 0.01
  # l2_reg_loss *= l2_lambda
  print_and_log('Task loss: %s' % (final_loss))
  # print_and_log('Task loss: %s, L2 regularization loss: %s' % (final_loss, l2_reg_loss))
  # Note: abs_error = np.sqrt(final_loss)
  # if final_loss < 0.005:
  #   import code; code.interact(local=dict(globals(), **locals()))
  return final_loss
  # return final_loss + l2_reg_loss

def L2_regularization(params):
  flat_params, _ = flatten(params)
  return np.sum(flat_params**2)
  
  # return np.sum(nonneg_transform(flat_params)**2)

def single_estimate(params, y, guide_row):
  [editor_params, batch_params, activity_params, background_param] = params
  
  pos_idx = pos_to_idx(guide_row['Position'])
  mut_idx = mut_to_idx(guide_row['Ref nt'], guide_row['Obs nt'])
  activity_idx = name_to_idx(guide_row['Name'])
  editor_idx = editor_to_idx(guide_row['Editor'])
  batch_idx = batch_to_idx(guide_row['Batch'])

  activity_p = nonneg_transform(activity_params[activity_idx])
  edit_p = nonneg_transform(editor_params[editor_idx][pos_idx][mut_idx])
  batch_p = nonneg_transform(batch_params[batch_idx][pos_idx][mut_idx])
  bg_p = nonneg_transform(background_param)

  # y_estimate = edit_p + batch_p + bg_p
  y_estimate = activity_p * edit_p + batch_p + bg_p
  return y_estimate

def learn_covariate_model(seed):
  # Load data
  print('Loading data...')
  covariates, Y_human = load_data()
  Y, guide = transform_to_machine_data(covariates, Y_human)
  '''
    guide columns:
      Name
      Batch
      Editor
      Position
      Ref nt
      Obs nt
      Mean activity
    All necessary information to determine which parameters to grab
  '''

  # Init parameters
  print('Initializing parameters...')
  params = init_params(guide)

  # Setup environment
  out_dir, out_letters, out_dir_params = setup_outdir()
  print(out_letters)
  print_and_log('seed: %s' % (seed))

  # Optimize parameters
  step_size = 0.1

  # batch_size = 1000
  # num_epochs = int(15.1 * (len(Y) / batch_size))
  batch_size = len(Y)
  # num_epochs = int(100.1 * (len(Y) / batch_size))
  
  # Overnight
  # Approx 1 epoch per minute
  # num_epochs = int(500.1 * (len(Y) / batch_size))
  num_epochs = int(1200.1 * (len(Y) / batch_size))


  num_batches = int(np.ceil(len(Y) / batch_size))

  from sklearn.utils import shuffle
  Y, guide = shuffle(Y, guide, random_state = rs)

  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  def objective(params, iter):
    slicer = batch_indices(iter)
    return main_objective(params, Y[slicer], guide.iloc[slicer])

  grad_fun = grad(objective)

  def print_perf(params, iter):
    letters = alphabetize(iter)
    print_and_log('Iter %s, %s' % (iter, letters))

    data = params_transform_to_human(params)
    with open(out_dir + 'data_%s.pkl' % (letters), 'wb') as f:
      pickle.dump(data, f)

    return None

  optimized_params = adam(grad_fun,
                          params, 
                          step_size = step_size, 
                          num_iters = num_epochs,
                          callback = print_perf)


  return
  

################################################################
# Starter
################################################################
##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for idx in range(10):
    command = 'python %s.py %s' % (NAME, idx)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, idx)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -l h_rt=20:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return


@util.time_dec
def main(seed = 0):
  print(NAME)

  global rs
  rs = npr.RandomState(int(seed))

  learn_covariate_model(int(seed))

  return

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(seed = sys.argv[1])
  else:
    gen_qsubs()