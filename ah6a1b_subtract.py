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

info_cols = ['Name', 'Category', 'Indel start', 'Indel end', 'Indel length', 'MH length', 'Inserted bases']

def subtract_treatment_control(t, c):
  c_cat_set = set(c['Category'])
  if len(c_cat_set) == 1 and 'wildtype' in c_cat_set:
    return t
  t_cat_set = set(t['Category'])
  if len(t_cat_set) == 1 and 'wildtype' in t_cat_set:
    return t

  mdf = t.merge(
    c, 
    how = 'outer',
    on = info_cols,
    suffixes = ['_t', '_c'],
  )

  mdf['Count_t'].fillna(value = 0, inplace = True)
  mdf['Count_c'].fillna(value = 0, inplace = True)

  c_tot = sum(mdf['Count_c'])
  t_tot = sum(mdf['Count_t'])

  dd = defaultdict(list)
  for idx, row in mdf.iterrows():
    if row['Category'] == 'wildtype':
      dd['Count_t_adj'].append(row['Count_t'])
    else:
      adj_val = row['Count_t'] - (row['Count_c'] / c_tot) * t_tot
      adj_val = max(0, adj_val)
      dd['Count_t_adj'].append(adj_val)

  for col in dd:
    mdf[col] = dd[col]

  adj_t = mdf
  adj_t['Count'] = adj_t['Count_t_adj']
  adj_t = adj_t[adj_t['Count'] > 0]
  adj_t = adj_t[info_cols + ['Count']]

  return adj_t

##
# Main iterator
##
def adjust_treatment_control(treat_nm, control_nm, start_idx, end_idx):
  adj_d = _data.load_data(treat_nm, 'ah6a1a_hf_bc')
  control_data = _data.load_data(control_nm, 'h6_anyindel')

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  lib_design = lib_design.iloc[start_idx : end_idx + 1]
  nms = lib_design['Name (unique)']

  '''
  '''
  print('Subtracting control from treatment data...')
  shared_nms = [nm for nm in nms if nm in adj_d]
  new_adj_d = dict()
  timer = util.Timer(total = len(shared_nms))
  for nm in shared_nms:
    t = adj_d[nm]
    if nm not in control_data:
      continue
    c = control_data[nm]

    t = subtract_treatment_control(t, c)
    new_adj_d[nm] = t
    timer.update()

  ##
  # Write
  ##
  with open(out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx), 'wb') as f:
    pickle.dump(new_adj_d, f)

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
    lib_nm = _data.get_lib_nm(treat_nm)


    if lib_nm == 'LibA':
      num_targets = 2000
      num_targets_per_split = 200
    elif lib_nm == 'CtoGA':
      num_targets = 4000
      num_targets_per_split = 500
    else:
      num_targets = 12000
      num_targets_per_split = 2000

    '''
      Empirically determined
      pickle > 37 mb: needs 4 gb ram
      pickle > 335 mb: needs 8 gb ram
    '''
    print(treat_nm)
    mb_file_size = _data.check_file_size(treat_nm, 'ah6a1a_hf_bc')
    ram_gb = 2
    if mb_file_size > 30:
      ram_gb = 4
    if mb_file_size > 300:
      ram_gb = 8
    if mb_file_size > 1000:
      ram_gb = 16

    '''
      Can be very slow - up to 8h+ for some conditions.

      Could help to split 3 steps into 3 scripts.
      Statistical tests should be performed globally (for accurate FDR thresholds), and luckily these are the fast parts of the pipeline

      Subtracting control from treatment involves a lot of dataframe manipulations and is the bottleneck step. Fortunately, this can be parallelized
    '''

    for start_idx in range(0, num_targets, num_targets_per_split):
      end_idx = start_idx + num_targets_per_split - 1

      out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx)
      if os.path.exists(out_pkl_fn):
        if os.path.getsize(out_pkl_fn) > 0:
          continue

      command = 'python %s.py %s %s %s %s' % (NAME, treat_nm, control_nm, start_idx, end_idx)
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s_%s.sh' % (script_id, treat_nm, control_nm, start_idx)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      num_scripts += 1

      # Write qsub commands
      qsub_commands.append('qsub -V -P regevlab -l h_rt=16:00:00,h_vmem=%sG -wd %s %s &' % (ram_gb, _config.SRC_DIR, sh_fn))

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
def main(treat_nm = '', control_nm = '', start_idx = '', end_idx = ''):
  print(NAME)
  
  # Function calls
  adjust_treatment_control(treat_nm, control_nm, int(start_idx), int(end_idx))

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(
      treat_nm = sys.argv[1], 
      control_nm = sys.argv[2],
      start_idx = sys.argv[3],
      end_idx = sys.argv[4],
    )
  else:
    gen_qsubs()
