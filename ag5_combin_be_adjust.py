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
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

##
# Functions
##
def get_significant_poswise_muts(df):
  dd = defaultdict(lambda: defaultdict(lambda: 0))
  nt_cols = [s for s in df.columns if s != 'Count']
  for idx, row in df.iterrows():
    count = row['Count']
    for nt_col in nt_cols:
      obs_nt = row[nt_col]
      if obs_nt != '.':
        dd[nt_col][obs_nt] += count

  sig_muts = []      
  for nt_col in dd:
    ref_nt = nt_col[0]
    tot = sum(dd[nt_col].values())
    for obs_nt in dd[nt_col]:
      if obs_nt == ref_nt:
        continue

      num = dd[nt_col][obs_nt]
      threshold = max(3, tot * 0.001 * 2)
      if num >= threshold:
        sig_muts.append('%s,%s' % (nt_col, obs_nt))

  return sig_muts, dd


def adjust_treatment_control(treat_nm, control_nm):
  treat_data = _data.load_data(treat_nm, 'g5_combin_be')
  control_data = _data.load_data(control_nm, 'g5_combin_be')

  lib_design, seq_col = _data.get_lib_design(treat_nm)

  ''' 
    g5 format: data is a dict, keys = target site names
    values = dfs, with columns as '%s%s' % (nt, position), and 'Count' column
  '''

  adj_d = dict()
  stats_dd = defaultdict(list)

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    timer.update()

    stats_dd['Name'].append(nm)

    if nm not in treat_data:
      stats_dd['Status'].append('No treatment')
      continue

    t = treat_data[nm]
    if nm not in control_data:
      stats_dd['Status'].append('No control')
      adj_d[nm] = t
      continue

    stats_dd['Status'].append('Adjusted')

    c = control_data[nm]
    if 'index' in t.columns:
      t = t.drop(columns = ['index'])
      c = c.drop(columns = ['index'])

    c = c[t.columns]

    nt_cols = [s for s in t.columns if s != 'Count']
    c = c.groupby(nt_cols)['Count'].sum()
    c = c.reset_index()

    # Detect columns with high mutation rate in control
    cm, cm_dd = get_significant_poswise_muts(c)
    tm, _ = get_significant_poswise_muts(t)

    bad_nt_cols = []
    for sig_mut in cm:
      if sig_mut in tm:
        [nt_col, obs_nt] = sig_mut.split(',')
        bad_nt_cols.append(nt_col)

    # Check corner case
    for nt_col in cm_dd:
      tot = sum(cm_dd[nt_col].values())
      ref_nt = nt_col[0]
      for obs_nt in cm_dd[nt_col]:
        if obs_nt != ref_nt:
          ct = cm_dd[nt_col][obs_nt]
          # if ct / tot > 0.01 and ct < max(3, tot * 0.001 * 2):
            # if '%s,%s' % (nt_col, obs_nt) in tm:
              # import code; code.interact(local=dict(globals(), **locals()))

    if len(bad_nt_cols) > 0:
      t = t[[col for col in t.columns if col not in bad_nt_cols]]
      c = c[[col for col in c.columns if col not in bad_nt_cols]]

      nt_cols = [s for s in t.columns if s != 'Count']
      if len(nt_cols) > 0:
        c = c.groupby(nt_cols)['Count'].sum().reset_index()
        t = t.groupby(nt_cols)['Count'].sum().reset_index()
      else:
        t = pd.DataFrame()
        continue


    # # Convert control to fractions
    # c['Count'] /= sum(c['Count'])

    # # Convert to readcounts using total treatment readcount as denominator  
    # c['Count'] *= sum(t['Count'])

    # # Zero out wild-type elements
    # pass

    # # Adjust treatment by control
    # c = c.set_index(nt_cols)
    # t = t.set_index(nt_cols)
    # t = t.subtract(c, fill_value = 0)

    # # Remove negative values
    # t = t[t['Count'] >= 0]

    t = t.sort_values(by = 'Count', ascending = False)
    t = t.reset_index()
    if 'index' in t.columns:
      t = t.drop(columns = ['index'])

    adj_d[nm] = t

  with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
    pickle.dump(adj_d, f)

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + '%s_stats.csv' % (treat_nm))

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
    treat_nm, control_nm = row['Treatment'], row['Control']

    if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
      continue
    if 'Cas9' in treat_nm:
      continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, control_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -l h_rt=4:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
def main(treat_nm = '', control_nm = ''):
  print(NAME)
  
  # Function calls
  adjust_treatment_control(treat_nm, control_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1], control_nm = sys.argv[2])
  else:
    gen_qsubs()
