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

# Default params
inp_dir = _config.OUT_PLACE
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)

##
# Support
##
def adjust_indel_pos(indel_start, indel_end, mh_len):
  '''
    seq_align provides right-aligned indels.
    mh_len describes maximum left shift possible while
    retaining alignment score.
    Find the left shift that minimizes the indel midpoint's
    distance to either the maximum base editing position
    or the HNH nick position
  '''
  # Note: this varies by editor (especially CP), but i'm not taking that into account for now
  max_baseedit_pos = 6
  hnh_pos = 18

  indel_len_temp = indel_end - indel_start

  best_dist = 1e4
  best_left_shift = 0
  for left_shift in range(int(mh_len)):
    mid_point = np.mean([indel_start - left_shift, indel_end - left_shift])

    if indel_len_temp < 7:
      # Shorter: consider only positions 6 and 18 as centers
      dist_to_be = abs(max_baseedit_pos - mid_point)
      dist_to_hnh = abs(hnh_pos - mid_point)
      dist = min(dist_to_be, dist_to_hnh)

      if dist < best_dist:
        best_dist = dist
        best_left_shift = left_shift
    else:
      # Longer, try to right-align on position 18
      dist_to_be = abs(max_baseedit_pos - mid_point)
      dist_to_hnh = abs(hnh_pos - mid_point)
      dist = min(dist_to_be, dist_to_hnh)

      if dist < best_dist:
        best_dist = dist
        best_left_shift = left_shift

      if indel_end - left_shift == hnh_pos:
        best_dist = dist
        best_left_shift = left_shift
        break

  adj_indel_start = indel_start - best_left_shift
  adj_indel_end = indel_end - best_left_shift

  return adj_indel_start, adj_indel_end

##
# Primary logic
##
def indel_anyindel_pos(exp_nm):
  data = None
  if exp_nm in set(treat_control_df['Treatment']):
    data = _data.load_data(exp_nm, 'ah6a3_remove_batch')
    is_control = False
  elif exp_nm in set(treat_control_df['Control']):
    data = _data.load_data(exp_nm, 'h6_anyindel')
    is_control = True

  if data is None:
    print('Error : could not load data')
    import code; code.interact(local=dict(globals(), **locals()))
    sys.exit(1)

  lib_design, seq_col = _data.get_lib_design(exp_nm)

  # Init
  pos_dd = dict()
  for pos_idx in range(-25, 50):
    pos_dd[pos_idx] = []
  pos_dd['Name'] = []

  # Init
  len_dd = dict()
  for len_val in range(-40, 15 + 1):
    len_dd[len_val] = []
  len_dd['Name'] = []

  # Init
  pos_len_indel = dict()
  for indel_len in range(-40, 15 + 1):
    pos_nt_indel = dict()
    for pos_idx in range(-25, 50):
      pos_nt_indel[pos_idx] = []
    pos_nt_indel['Name'] = []
    pos_len_indel[indel_len] = pos_nt_indel

  mdf = pd.DataFrame()
  timer = util.Timer(total = len(data))
  for target_nm in data:
    df = data[target_nm]

    if 'Frequency' not in df.columns:
      df['Frequency'] = df['Count'] / np.sum(df['Count'])

    tot_count = sum(df['Count'])
    if tot_count < 100:
      continue

    crit = (df['Category'] != 'wildtype')
    dfs = df[crit]
    

    # Init
    target_pos_vector = defaultdict(lambda: 0)
    target_pos_len_vectors = dict()
    for indel_len in range(-40, 15 + 1):
      target_pos_vector_nt = defaultdict(lambda: 0)
      target_pos_len_vectors[indel_len] = target_pos_vector_nt
    indel_len_vector = defaultdict(lambda: 0)

    # Iterate
    df_annot = defaultdict(list)
    for idx, row in dfs.iterrows():
      indel_start = int(row['Indel start'])
      indel_end = int(row['Indel end'])
      mh_len = row['MH length']
      freq = row['Frequency']
      indel_len = int(row['Indel length'])
      cat = row['Category']

      # Gather indel length frequencies
      if cat == 'del':
        indel_len = indel_len * -1
      elif cat == 'ins':
        '''
          h6_anyindel in each dir describes indel_start and indel_end for indexing (should have difference of 1 for insertions but they do not). so we fix this here
        '''
        indel_len = indel_len
        indel_end = indel_start + 1

      indel_len_vector[indel_len] += freq

      # Adjust start and end
      if not is_control:
        package = adjust_indel_pos(indel_start, indel_end, mh_len)
        (adj_indel_start, adj_indel_end) = package
      else:
        adj_indel_start, adj_indel_end = indel_start, indel_end

      # Gather total frequency by position
      for jdx in range(adj_indel_start, adj_indel_end):
        target_pos_vector[jdx] += freq

      # Gather total frequency by position of specific nt indels
      if indel_len in target_pos_len_vectors:
        target_pos_vector_nt = target_pos_len_vectors[indel_len]
        for jdx in range(adj_indel_start, adj_indel_end):
          target_pos_vector_nt[jdx] += freq

      df_annot['Indel start adj'].append(adj_indel_start)
      df_annot['Indel end adj'].append(adj_indel_end)

    for col in df_annot:
      dfs[col] = df_annot[col]
    dfs['Name'] = target_nm
    mdf = mdf.append(dfs, ignore_index = True)

    # Gather indel length frequencies
    for col in len_dd:
      if col != 'Name':
        len_dd[col].append(indel_len_vector[col])
      else:
        len_dd[col].append(target_nm)

    # Gather total frequency by position
    for col in pos_dd:
      if col != 'Name':
        pos_dd[col].append(target_pos_vector[col])
      else:
        pos_dd[col].append(target_nm)

      # Gather total frequency by position of 1 nt indels
    for indel_len in pos_len_indel:
      pos_nt_indel = pos_len_indel[indel_len]
      tpvn = target_pos_len_vectors[indel_len]

      for col in pos_nt_indel:
        if col != 'Name':
          pos_nt_indel[col].append(tpvn[col])
        else:
          pos_nt_indel[col].append(target_nm)

    timer.update()

  # Save
  pos_df = pd.DataFrame(pos_dd)
  pos_df.to_csv(out_dir + '%s_pos.csv' % (exp_nm))

  pos_df_melt = pd.melt(pos_df, id_vars = 'Name', var_name = 'Position', value_name = 'Frequency')
  pos_df_melt.to_csv(out_dir + '%s_pos_melt.csv' % (exp_nm))

  # Save
  for indel_len in pos_len_indel:
    pos_nt_indel = pos_len_indel[indel_len]

    pos_nt_df = pd.DataFrame(pos_nt_indel)
    pos_nt_df.to_csv(out_dir + '%s_pos_%snt.csv' % (exp_nm, indel_len))

    pos_nt_df_melt = pd.melt(pos_nt_df, id_vars = 'Name', var_name = 'Position', value_name = 'Frequency')
    pos_nt_df_melt.to_csv(out_dir + '%s_pos_melt_%snt.csv' % (exp_nm, indel_len))

  # Save
  len_df = pd.DataFrame(len_dd)
  len_df.to_csv(out_dir + '%s_len.csv' % (exp_nm))

  len_df_melt = pd.melt(len_df, id_vars = 'Name', var_name = 'Indel length', value_name = 'Frequency')
  len_df_melt.to_csv(out_dir + '%s_len_melt.csv' % (exp_nm))

  # merged
  mdf.to_csv(out_dir + '%s.csv' % (exp_nm))
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

  treatments = list(set(treat_control_df['Treatment']))
  controls = list(set(treat_control_df['Control']))

  all_conds = treatments
  # all_conds = treatments + controls
  # all_conds = controls

  num_scripts = 0
  for cond in all_conds:

    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, cond)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=6:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  
  indel_anyindel_pos(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()