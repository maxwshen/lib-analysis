# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

##
# Functions
##
def build_new_cas(lib, cas):
  # Find shared indel events and place into a single DF.
  # Also use shared indel events to adjust counts in treatment
  lib_total = sum(lib['Count'])
  cas_total = sum(cas['Count'])
  if lib_total < 1000 or cas_total == 0:
    # print(lib_total)
    return cas

  join_cols = list(cas.columns)
  join_cols.remove('_ExpDir')
  join_cols.remove('_Experiment')
  join_cols.remove('_Cutsite')
  join_cols.remove('_Sequence Context')
  join_cols.remove('Count')

  mdf = cas.merge(lib, how = 'outer', on = join_cols, suffixes = ('', '_control'))

  mdf['Count_control'].fillna(value = 0, inplace = True)
  mdf['Count'].fillna(value = 0, inplace = True)
  
  total_control = sum(mdf['Count_control'])
  total_treatment = sum(mdf['Count'])

  def control_adjust(row):
    ctrl_frac = row['Count_control'] / total_control
    treat_frac = row['Count'] / total_treatment
    if treat_frac == 0:
      return 0
    adjust_frac = (treat_frac - ctrl_frac) / treat_frac
    ans = max(row['Count'] * adjust_frac, 0)
    return ans

  new_counts = mdf.apply(control_adjust, axis = 1)
  mdf['Count'] = new_counts

  mdf = mdf[mdf['Count'] != 0]
  for col in mdf.columns:
    if '_control' in col:
      mdf = mdf.drop(labels = col, axis = 1)

  return mdf

##
# Iterator
##
def adjust_treatment_control(treat_nm, control_nm):
  cas_data = _data.load_data(treat_nm, 'e_newgenotype_Cas9')
  lib_data = _data.load_data(control_nm, 'e_newgenotype_Cas9')
  cas_data = cas_data.drop(columns = 'ULMI count')
  lib_data = lib_data.drop(columns = 'ULMI count')

  lib_design, seq_col = _data.get_lib_design(treat_nm)

  adj_d = dict()
  stats_dd = defaultdict(list)

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    timer.update()

    cas = cas_data[cas_data['_Experiment'] == nm]
    lib = lib_data[lib_data['_Experiment'] == nm]

    stats_dd['Name'].append(nm)
    if len(cas) == 0:
      stats_dd['Status'].append('No treatment')
      continue
    if len(lib) == 0:
      stats_dd['Status'].append('No control')
      adj_d[nm] = t
      continue

    stats_dd['Status'].append('Adjusted')
    new_cas = build_new_cas(lib, cas)
    adj_d[nm] = new_cas

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
    if 'Cas9' not in treat_nm:
      continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, control_nm)
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
