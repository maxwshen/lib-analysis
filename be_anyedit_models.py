# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Default params
inp_dir = _config.OUT_PLACE + 'be_combin_general_annot/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Statistics
##
def train_models(exp_nm, data, ml_task, seq_col):
  # Prepare models and data

  if ml_task == 'regress_nonzero':
    models = {
      'LinearReg': Ridge(),
      'GBTR': GradientBoostingRegressor(),
    }
    data['y'] = data['Fraction edited']
    evals = {
      'spearmanr': lambda t, p: spearmanr(t, p)[0],
      'pearsonr': lambda t, p: pearsonr(t, p)[0],
      'r2_score': lambda t, p: sklearn.metrics.r2_score(t, p),
    }

  # One hot encode
  encoder = sklearn.preprocessing.OneHotEncoder()
  d = []
  for idx, row in data.iterrows():
    x_input = row[seq_col]
    zero_idx = _data.pos_to_idx(0, exp_nm)
    x_input = x_input[zero_idx - 9 : zero_idx + 20 + 1]
    d.append( list(x_input) )
  X_all = encoder.fit_transform(d).toarray()

  # Train test split
  package = train_test_split(
    X_all, 
    np.array(data['y']), 
    list(data['Name (unique)']),
  )
  (x_train, x_test, y_train, y_test, nms_train, nms_test) = package

  # Train models
  ms_dd = defaultdict(list)
  ms_dd['Name'].append(exp_nm)

  for model_nm in models:
    print(model_nm)
    model = models[model_nm]

    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    # Store model performance stats in modelstats_dd
    for ml_eval_nm in evals:
      eval_f = evals[ml_eval_nm]

      try:
        ev = eval_f(y_train, pred_train)
      except ValueError:
        ev = np.nan
      ms_dd['%s %s train' % (model_nm, ml_eval_nm)].append(ev)

      try:
        ev = eval_f(y_test, pred_test)
      except ValueError:
        ev = np.nan
      ms_dd['%s %s test' % (model_nm, ml_eval_nm)].append(ev)

    # Store regression parameters in modelstats_dd
    if 'Reg' in model_nm:
      params = model.coef_.reshape(-1)
      param_nms = encoder.get_feature_names()
      for val, nm in zip(params, param_nms):
        ms_dd[nm].append(val)

    # Record predictions in data
    pred_df = pd.DataFrame({
      'Name (unique)': nms_train + nms_test,
      'y_pred_%s' % (model_nm): list(pred_train) + list(pred_test),
      'TrainTest_%s' % (model_nm): ['train'] * len(nms_train) + ['test'] * len(nms_test)
    })
    data = data.merge(pred_df, on = 'Name (unique)')

  ms_df = pd.DataFrame(ms_dd)
  ms_df = ms_df.reindex(sorted(ms_df.columns), axis = 1)
  return (ms_df, data)

##
# IO
##
def save_results(exp_nm, ml_task, results):
  (model_stats, pred_obs_df) = results

  model_stats.to_csv(out_dir + '%s_%s_model_stats.csv' % (exp_nm, ml_task))

  pred_obs_df.to_csv(out_dir + '%s_%s_df.csv' % (exp_nm, ml_task))

  return

##
# Primary logic
##
def gather_statistics(exp_nm):
  # Load data
  data = pd.read_csv(inp_dir + '%s.csv' % (exp_nm), index_col = 0)

  # Prepare data
  data = data[data['Total count'] >= 100]
  lib_nm = _data.get_lib_nm(exp_nm)
  lib_design, seq_col = _data.get_lib_design(exp_nm)
  ontarget_sites = _data.get_ontarget_sites(lib_design, lib_nm)
  data = data[data['Name (unique)'].isin(ontarget_sites)]

  # # Gather statistics
  for ml_task in ['regress_nonzero']:
    print(ml_task)
    results = train_models(exp_nm, data, ml_task, seq_col)
    save_results(exp_nm, ml_task, results)

  return



##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  # Generate qsubs only for unfinished jobs
  treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  num_scripts = 0
  for idx, row in treat_control_df.iterrows():
    treat_nm = row['Treatment']
    if 'Cas9' in treat_nm:
      continue

    command = 'python %s.py %s' % (NAME, treat_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, treat_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return

##
# Main
##
@util.time_dec
def main(exp_nm = ''):
  print(NAME)
  
  # # Forward strand 
  gather_statistics(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()