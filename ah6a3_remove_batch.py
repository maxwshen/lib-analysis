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
inp_dir = _config.OUT_PLACE + 'ah6a2_anyindel_batch/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
batches = exp_design['Batch covariate (indels)']
editors = exp_design['Editor']
exp_nm_to_batch = {exp_nm: batch for exp_nm, batch in zip(exp_nms, batches)}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

info_cols = ['Name', 'Category', 'Indel start', 'Indel end', 'Indel length', 'MH length', 'Inserted bases']

##
# Filters
##
def filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm):
  new_adj_d = dict()

  shared_nms = [nm for nm in adj_d if nm in nm_to_seq]
  timer = util.Timer(total = len(shared_nms))
  for exp_nm in shared_nms:

    t = adj_d[exp_nm]
    tr = to_remove[to_remove['Name'] == exp_nm]

    for idx, row in tr.iterrows():
      mut_nm = row['MutName']
      cat, ids, ide, idl, mhl, ib = mut_nm.split('_')

      t_cat_set = set(t['Category'])
      if len(t_cat_set) == 1 and 'wildtype' in t_cat_set:
        new_adj_d[exp_nm] = t
        continue

      crit = (t['Category'] == cat) & (t['Indel start'] == float(ids)) & (t['Indel end'] == float(ide)) & (t['Indel length'] == float(idl)) & (t['MH length'] == float(mhl)) & (t['Inserted bases'] == ib)
      t.loc[crit, 'Count'] = 0

    t = t[t['Count'] > 0]
    t['Frequency'] = t['Count'] / sum(t['Count'])
    new_adj_d[exp_nm] = t

    timer.update()

  return new_adj_d

##
# Main iterator
##
def remove_batch_effects(treat_nm, start_idx, end_idx):
  batch_nm = exp_nm_to_batch[treat_nm]

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  lib_nm = _data.get_lib_nm(treat_nm)
  lib_design = lib_design.iloc[start_idx : end_idx + 1]
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  adj_d = _data.load_data(treat_nm, 'ah6a1b_subtract')

  batch_muts_to_remove = pd.read_csv(inp_dir + 'removed_batch_effects_%s.csv' % (lib_nm), index_col = 0)

  if len(batch_muts_to_remove) == 0:
    inp_pkl = _config.OUT_PLACE + f'ah6a1b_subtract/{treat_nm}_{start_idx}_{end_idx}.pkl'
    out_pkl = out_dir + f'{treat_nm}_{start_idx}_{end_idx}.pkl'
    command = f'cp {inp_pkl} {out_pkl}'
    subprocess.check_output(command, shell = True)
    return

  # Remove mutations
  to_remove = batch_muts_to_remove[batch_muts_to_remove['Batch'] == batch_nm]
  to_remove = to_remove[to_remove['Name'].isin(nms)]
  
  adj_d = filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm)

  with open(out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx), 'wb') as f:
    pickle.dump(adj_d, f)

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


    mb_file_size = _data.check_file_size(treat_nm, 'ah6a1b_subtract')
    ram_gb = 2
    if mb_file_size > 30:
      ram_gb = 4
    if mb_file_size > 300:
      ram_gb = 8
    if mb_file_size > 1000:
      ram_gb = 16

    for start_idx in range(0, num_targets, num_targets_per_split):
      end_idx = start_idx + num_targets_per_split - 1

      out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx)
      if os.path.exists(out_pkl_fn):
        if os.path.getsize(out_pkl_fn) > 0:
          continue

      command = 'python %s.py %s %s %s' % (NAME, treat_nm, start_idx, end_idx)
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, start_idx)
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
def main(treat_nm = '', start_idx = '', end_idx = ''):
  print(NAME)
  
  # Function calls
  remove_batch_effects(treat_nm, int(start_idx), int(end_idx))

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1], start_idx = sys.argv[2], end_idx = sys.argv[3])
  else:
    gen_qsubs()