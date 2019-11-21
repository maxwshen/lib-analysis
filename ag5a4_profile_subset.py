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
# inp_dir = _config.OUT_PLACE + 'ag5_combin_be_adjust/'
inp_dir = _config.OUT_PLACE + 'ag5a3_adjust_inprofile_batch_effects/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
editors = exp_design['Editor']
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

profile_design = pd.read_csv(_config.DATA_DIR + 'editor_profiles.csv')

def get_editor_mutations(editor_nm):
  row = profile_design[profile_design['Editor name'] == editor_nm].iloc[0]

  muts_dict = defaultdict(list)
  mut_cols = [col for col in profile_design if col != 'Editor name']
  for col in mut_cols:
    if type(row[col]) != str:
      continue

    [ref_nt, obs_nt] = col.replace(' to ', ' ').split()
    [start_pos, end_pos] = row[col].replace('(', '').replace(')', '').replace(',', '').split()
    start_pos, end_pos = int(start_pos), int(end_pos)

    for pos in range(start_pos, end_pos + 1):
      mut_col_nm = '%s%s' % (ref_nt, pos)
      muts_dict[mut_col_nm].append(obs_nt)
  return muts_dict


def profile_subset(exp_nm, start_idx, end_idx):
  editor_nm = exp_nm_to_editor[exp_nm]

  muts_dict = get_editor_mutations(editor_nm)
  editor_mut_set = set(muts_dict.keys())

  '''
    Identify batch effects from position-wise analysis and remove them from combinatorial df
  '''
  lib_design, seq_col = _data.get_lib_design(exp_nm)
  lib_design = lib_design.iloc[start_idx : end_idx + 1]
  nms = lib_design['Name (unique)']

  data = pickle.load(open(inp_dir + '%s.pkl' % (exp_nm), 'rb'))
  new_data = dict()

  nms_shared = [nm for nm in nms if nm in data]
  timer = util.Timer(total = len(nms_shared))
  for target_nm in nms_shared:
    d = data[target_nm]
    d = d[d['Count'] > 0]

    if 'index' in d.columns:
      d = d[[col for col in d.columns if col != 'index']]

    # Subset to mutation columns
    _mut_cols = [col for col in d.columns if col in editor_mut_set]
    d = d[_mut_cols + ['Count']]

    for idx, row in d.iterrows():
      for col in _mut_cols:
        ref_nt = col[0]
        obs_nt = row[col]
        if obs_nt != ref_nt and obs_nt not in muts_dict[col]:
          row[col] = '.'

    # Eliminate rows that only contain .
    crit = d.apply(
      lambda row: sum([row[c] == '.' for c in row.index]) != len(row.index) - 1, 
      axis = 'columns'
    )
    d = d[crit]

    d = d.reset_index(drop = True)

    new_data[target_nm] = d
    timer.update()

  with open(out_dir + '%s_%s_%s.pkl' % (exp_nm, start_idx, end_idx), 'wb') as f:
    pickle.dump(new_data, f)

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
    lib_nm = _data.get_lib_nm(exp_nm)

    if 'Cas9' in exp_nm:
      continue

    if lib_nm == 'LibA':
      num_targets = 2000
      num_targets_per_split = 200
    elif lib_nm == 'CtoGA':
      num_targets = 4000
      num_targets_per_split = 500
    else:
      num_targets = 12000
      num_targets_per_split = 2000

    try:
      mb_file_size = os.path.getsize(inp_dir + '%s.pkl' % (exp_nm)) / 1e6
    except FileNotFoundError:
      mb_file_size = 0
    ram_gb = 2
    if mb_file_size > 200:
      ram_gb = 4
    if mb_file_size > 400:
      ram_gb = 8
    if mb_file_size > 1000:
      ram_gb = 16


    for start_idx in range(0, num_targets, num_targets_per_split):
      end_idx = start_idx + num_targets_per_split - 1

      out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (exp_nm, start_idx, end_idx)
      if os.path.exists(out_pkl_fn):
        if os.path.getsize(out_pkl_fn) > 0:
          continue

      command = 'python %s.py %s %s %s' % (NAME, exp_nm, start_idx, end_idx)
      script_id = NAME.split('_')[0]


      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, exp_nm, start_idx)
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
def main(exp_nm = '', start_idx = '', end_idx = ''):
  print(NAME)
  
  # Function calls
  profile_subset(exp_nm, int(start_idx), int(end_idx))

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1], start_idx = sys.argv[2], end_idx = sys.argv[3])
  else:
    gen_qsubs()
