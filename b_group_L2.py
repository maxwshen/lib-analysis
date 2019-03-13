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

group_design = pd.read_csv(_config.DATA_DIR + 'master_L2_groups.csv', index_col = 0)


##
# Form groups
##
def form_L2_group_ag4_poswise_be_adjust(group_nm, l1_nms):
  datas = [_data.load_data(l1_nm, 'ag4_poswise_be_adjust') for l1_nm in l1_nms]
  datas = [s for s in datas if s is not None]
  lib_design, seq_col = _data.get_lib_design(l1_nms[0])
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''
  group_ds = dict()

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    timer.update()

    # num_present = sum([bool(nm in data) for data in datas])

    '''
    Combine: two strategies
    1. Normalize readcount, then add (equal contribution)
    2. Directly add (weighted by readcount)
      * using this strategy
    '''
    group_d = None
    for data in datas:

      if nm not in data:
        continue
      d = data[nm]

      if group_d is None:
        group_d = d
      else:
        group_d += d

    if group_d is not None:
      group_ds[nm] = group_d

  with open(out_dir + '%s.pkl' % (group_nm), 'wb') as f:
    pickle.dump(group_ds, f)

  return

def form_L2_group_ag5_combin_be_adjust(group_nm, l1_nms):
  datas = [_data.load_data(l1_nm, 'ag5_combin_be_adjust') for l1_nm in l1_nms]
  datas = [s for s in datas if s is not None]
  lib_design, seq_col = _data.get_lib_design(l1_nms[0])
  ''' 
    g5 format: data is a dict, keys = target site names
    values = dfs, with columns as '%s%s' % (nt, position), and 'Count' column
  '''
  group_ds = dict()

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    timer.update()

    # num_present = sum([bool(nm in data) for data in datas])

    '''
    Combine: two strategies
    1. Normalize readcount, then add (equal contribution)
    2. Directly add (weighted by readcount)
      * using this strategy
    '''
    group_d = None
    for data in datas:

      if nm not in data:
        continue

      d = data[nm]
      nt_cols = [s for s in d.columns if s != 'Count']

      if group_d is None:
        group_d = d
      else:
        group_d = group_d.append(d, ignore_index = True)
        group_d = group_d.groupby(nt_cols)['Count'].sum()
        group_d = group_d.reset_index()

    if group_d is not None:
      group_d = group_d.sort_values(by = 'Count', ascending = False)
      group_d = group_d.reset_index()
      group_ds[nm] = group_d

  with open(out_dir + '%s.pkl' % (group_nm), 'wb') as f:
    pickle.dump(group_ds, f)

  return

def form_L2_group_ae_newgenotype_Cas9_adjust(group_nm, l1_nms):
  datas = [_data.load_data(l1_nm, 'ae_newgenotype_Cas9_adjust') for l1_nm in l1_nms]
  datas = [s for s in datas if s is not None]
  lib_design, seq_col = _data.get_lib_design(l1_nms[0])
  ''' 
    g5 format: data is a dict, keys = target site names
    values = dfs, with columns as '%s%s' % (nt, position), and 'Count' column
  '''
  group_ds = dict()

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    timer.update()

    # num_present = sum([bool(nm in data) for data in datas])

    '''
    Combine: two strategies
    1. Normalize readcount, then add (equal contribution)
    2. Directly add (weighted by readcount)
      * using this strategy
    '''
    group_d = None
    for data in datas:

      if nm not in data:
        continue

      d = data[nm]
      d = d.drop(columns = ['_Sequence Context', '_Cutsite', '_ExpDir'])
      nt_cols = [s for s in d.columns if s != 'Count']
      d = d.fillna(value = '.')

      if group_d is None:
        group_d = d
      else:
        group_d = group_d.append(d, ignore_index = True)
        group_d = group_d.groupby(nt_cols)['Count'].sum()
        group_d = group_d.reset_index()

    if group_d is not None:
      group_d = group_d.sort_values(by = 'Count', ascending = False)
      group_d = group_d.reset_index()
      group_d = group_d.replace(to_replace = '.', value = np.nan)
      group_ds[nm] = group_d

  with open(out_dir + '%s.pkl' % (group_nm), 'wb') as f:
    pickle.dump(group_ds, f)

  return



##
# Iterator
##
def form_L2_group(group_nm):
  row = group_design[group_design['Name'] == group_nm].iloc[0]
  item_cols = [s for s in group_design.columns if 'Item' in s]
  l1_nms = [row[item_col] for item_col in item_cols]
  l1_nms = [s for s in l1_nms if type(s) != float]

  if 'UT' in group_nm:
    scripts = ['ag4_poswise_be_adjust', 'ag5_combin_be_adjust', 'ae_newgenotype_Cas9_adjust']
  elif 'Cas9' in group_nm:
    scripts = ['ae_newgenotype_Cas9_adjust']
  else:
    scripts = ['ag4_poswise_be_adjust', 'ag5_combin_be_adjust']

  for script in scripts:
    print(script)
    if script == 'ag4_poswise_be_adjust':
      form_L2_group_ag4_poswise_be_adjust(group_nm, l1_nms)
    if script == 'ag5_combin_be_adjust':
      form_L2_group_ag5_combin_be_adjust(group_nm, l1_nms)
    if script == 'ae_newgenotype_Cas9_adjust':
      form_L2_group_ae_newgenotype_Cas9_adjust(group_nm, l1_nms)

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
  group_design = pd.read_csv(_config.DATA_DIR + 'master_L2_groups.csv', index_col = 0)

  num_scripts = 0
  for idx, row in group_design.iterrows():
    group_nm = row['Name']

    if row['Use flag'] != 1:
      continue

    command = 'python %s.py %s' % (NAME, group_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, group_nm)
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
def main(group_nm = ''):
  print(NAME)
  
  # Function calls
  if group_nm == '':
    for idx, row in group_design.iterrows():
      group_nm = row['Name']
      print(group_nm)
      form_L2_group(group_nm)

  else:
    form_L2_group(group_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(group_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()