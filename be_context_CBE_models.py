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
inp_dir = _config.OUT_PLACE + 'c_poswise_basic/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Statistics
##
def train_models(exp_nm, data, mut, ml_task):
  # Prepare models and data

  if ml_task == 'classify_zero':
    models = {
      'LogisticReg': LogisticRegression(),
      'GBTC': GradientBoostingClassifier(),
    }
    data['y'] = (data['Frequency'] > 0).astype(int)

    evals = {
      'AUC': lambda t, p: sklearn.metrics.roc_auc_score(t, p),
      'AveragePrecisionScore': lambda t, p: sklearn.metrics.average_precision_score(t, p),
      'CrossEntropy': lambda t, p: sklearn.metrics.log_loss(t, p),
    }

  if ml_task == 'regress_nonzero':
    models = {
      'LinearReg': Ridge(),
      'GBTR': GradientBoostingRegressor(),
    }
    data = data[data['Frequency'] > 0]
    data['y'] = data['Frequency']
    evals = {
      'spearmanr': lambda t, p: spearmanr(t, p)[0],
      'pearsonr': lambda t, p: pearsonr(t, p)[0],
      'r2_score': lambda t, p: sklearn.metrics.r2_score(t, p),
    }

  # One hot encode
  encoder = sklearn.preprocessing.OneHotEncoder()
  d = []
  for idx, row in data.iterrows():
    d.append( list(row['Local context']) )
  X_all = encoder.fit_transform(d).toarray()

  # Normalize y by position
  y_means = {pos: np.mean(data[data['Position'] == pos]['y']) for pos in set(data['Position'])}
  y_stds = {pos: np.std(data[data['Position'] == pos]['y']) for pos in set(data['Position'])}
  new_y = []
  for idx, row in data.iterrows():
    rowpos = row['Position']
    adj_y = (row['y'] - y_means[rowpos]) / y_stds[rowpos]
    new_y.append(adj_y)
  data['y'] = new_y

  # Train test split
  package = train_test_split(
    X_all, 
    np.array(data['y']), 
    list(data['MutName']),
  )
  (x_train, x_test, y_train, y_test, nms_train, nms_test) = package

  # Train models
  ms_dd = defaultdict(list)
  ms_dd['Name'].append(exp_nm)
  ms_dd['Mutation'].append(mut)

  for model_nm in models:
    print(model_nm)
    model = models[model_nm]

    model.fit(x_train, y_train)

    if ml_task == 'regress_nonzero':
      pred_train = model.predict(x_train)
      pred_test = model.predict(x_test)
    elif ml_task == 'classify_zero':
      # Get probability of y = 1
      pred_train = model.predict_proba(x_train).T[1]
      pred_test = model.predict_proba(x_test).T[1]

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
      'MutName': nms_train + nms_test,
      'y_pred_%s' % (model_nm): list(pred_train) + list(pred_test),
      'TrainTest_%s' % (model_nm): ['train'] * len(nms_train) + ['test'] * len(nms_test)
    })
    data = data.merge(pred_df, on = 'MutName')

  ms_df = pd.DataFrame(ms_dd)
  ms_df = ms_df.reindex(sorted(ms_df.columns), axis = 1)
  return (ms_df, data)

##
# IO
##
def save_results(exp_nm, mut, ml_task, results):
  (model_stats, pred_obs_df) = results

  model_stats.to_csv(out_dir + '%s_%s_%s_model_stats.csv' % (exp_nm, mut, ml_task))

  pred_obs_df.to_csv(out_dir + '%s_%s_%s_df.csv' % (exp_nm, mut, ml_task))

  return

##
# Primary logic
##
def gather_statistics(exp_nm, params):
  (muts, allowed_pos, feature_radius) = params
  # Load data
  data = pd.read_csv(inp_dir + '%s.csv' % (exp_nm), index_col = 0)

  # Set up library info
  lib_nm = _data.get_lib_nm(exp_nm)
  lib_design, seq_col = _data.get_lib_design(exp_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  # Prepare data
  data = data[data['Total count'] >= 100]
  data['Frequency'] = data['Count'] / data['Total count']

  ontarget_sites = _data.get_ontarget_sites(lib_design, lib_nm)
  data = data[data['Name'].isin(ontarget_sites)]

  data = data[data['Position'].isin(allowed_pos)]

  data['Mutation'] = data['Ref nt'] + '_' + data['Obs nt']
  data['MutName'] = data['Name'].astype(str) + '_' + data['Position'].astype(str) + '_' + data['Mutation']

  # Annotate with local sequence context
  lib_zero_idx = _data.pos_to_idx(0, exp_nm)
  dd = defaultdict(list)
  print('Annotating data with local sequence contexts...')
  timer = util.Timer(total = len(data))
  for idx, row in data.iterrows():
    seq = nm_to_seq[row['Name']]
    pidx = row['Position'] + lib_zero_idx
    local_context = seq[pidx - feature_radius : pidx] + seq[pidx + 1 : pidx + feature_radius + 1]
    dd['Local context'].append(local_context)
    timer.update()
  for col in dd:
    data[col] = dd[col]

  # # Gather statistics

  for mut_nm in muts:
    print(mut_nm)
    mut = muts[mut_nm]
    if len(mut) == 1:
      d_temp = data[data['Mutation'] == mut[0]]
    else:
      d_temp = data[data['Mutation'].isin(mut)]
      d_temp['Mutation'] = mut_nm
      d_temp['MutName'] = d_temp['Name'].astype(str) + '_' + d_temp['Position'].astype(str) + '_' + d_temp['Mutation']
      group_cols = [s for s in d_temp.columns if s not in ['Frequency', 'Obs nt', 'Ref nt', 'Count']]
      d_temp = d_temp.groupby(group_cols)['Frequency'].agg('sum').reset_index()

    for ml_task in ['regress_nonzero']:
      print(ml_task)
      results = train_models(exp_nm, d_temp, mut_nm, ml_task)
      save_results(exp_nm, mut_nm, ml_task, results)



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
    if 'ABE' in treat_nm:
      continue
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
    qsub_commands.append('qsub -V -l h_rt=4:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  
  # Forward strand 
  muts = {
    'C_G': ['C_G'],
    'C_A': ['C_A'],
    'C_GA': ['C_G', 'C_A'],
    'C_T': ['C_T'],
    'C_TGA': ['C_T', 'C_G', 'C_A'],
  }
  allowed_pos = list(range(3, 8 + 1))
  feature_radius = 10
  params = (muts, allowed_pos, feature_radius)
  gather_statistics(exp_nm, params)

  # Opposite strand
  muts = {
    'G_A': ['G_A'],
    'G_ACT': ['G_A', 'G_C', 'G_T'],
  }
  allowed_pos = list(range(-4, 0 + 1))
  feature_radius = 5
  params = (muts, allowed_pos, feature_radius)
  gather_statistics(exp_nm, params)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()