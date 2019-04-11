# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
from scipy.stats import binom

# Default params
inp_dir = _config.OUT_PLACE + 'ag5_combin_be_adjust/'
batch_dir = _config.OUT_PLACE + 'ag4a2_adjust_batch_effects/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
batches = exp_design['Batch covariate']
editors = exp_design['Editor']
exp_nm_to_batch = {exp_nm: batch for exp_nm, batch in zip(exp_nms, batches)}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

batch_to_exp_nm = defaultdict(list)
for exp_nm in exp_nm_to_batch:
  batch = exp_nm_to_batch[exp_nm]
  batch_to_exp_nm[batch].append(exp_nm)


def adjust_all_batch_effects():
  '''
    Identify batch effects from position-wise analysis and remove them from combinatorial df
  '''

  batch_muts = pd.read_csv(batch_dir + 'removed_batch_effects.csv')
  batches = sorted(set(batch_muts['Batch']))

  for batch in batches:
    print(batch)
    df = batch_muts[batch_muts['Batch'] == batch]

    batch_cols = []
    for idx, row in df.iterrows():
      batch_col = '%s%s' % (row['Ref nt'], row['Position'])

    exp_nms = batch_to_exp_nm[batch]
    timer = util.Timer(total = len(exp_nms))
    for exp_nm in exp_nms:
      data = pickle.load(open(inp_dir + '%s.pkl' % (exp_nm), 'rb'))

      for target_nm in data:
        d = data[target_nm]

        for batch_col in batch_cols:
          if batch_col in d.columns:
            d[batch_col] = '.'

        data[target_nm] = d

      with open(out_dir + '%s.pkl' % (exp_nm), 'wb') as f:
        pickle.dump(data, f)

      timer.update()

  return

def adjust_batch_effects(exp_nm):
  '''
    Identify batch effects from position-wise analysis and remove them from combinatorial df
  '''

  batch_muts = pd.read_csv(batch_dir + 'removed_batch_effects.csv')
  batch = exp_nm_to_batch[exp_nm]
  df = batch_muts[batch_muts['Batch'] == batch]

  batch_cols, obs_nts = [], []
  for idx, row in df.iterrows():
    batch_col = '%s%s' % (row['Ref nt'], row['Position'])
    batch_cols.append(batch_col)
    obs_nts.append(row['Obs nt'])

  data = pickle.load(open(inp_dir + '%s.pkl' % (exp_nm), 'rb'))

  timer = util.Timer(total = len(data))
  for target_nm in data:
    d = data[target_nm]

    for batch_col, obs_nt in zip(batch_cols, obs_nts):
      if batch_col in d.columns:
        d.loc[d[batch_col] == obs_nt, batch_col] = '.'

    data[target_nm] = d
    timer.update()

  with open(out_dir + '%s.pkl' % (exp_nm), 'wb') as f:
    pickle.dump(data, f)

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
  num_scripts = 0
  for idx, row in treat_control_df.iterrows():
    exp_nm = row['Treatment']

    if os.path.exists(out_dir + '%s.pkl' % (exp_nm)):
      continue
    if 'Cas9' in exp_nm:
      continue

    command = 'python %s.py %s' % (NAME, exp_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, exp_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -l h_rt=16:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  
  # Function calls
  adjust_batch_effects(exp_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
