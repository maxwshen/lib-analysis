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
inp_dir = _config.OUT_PLACE + 'ag5a4_profile_subset/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Statistics
##
def get_num_edits(row, col_to_ref_nt):
  num_edits = 0
  for col in col_to_ref_nt:
    ref_nt = col_to_ref_nt[col]
    obs_nt = row[col]
    if obs_nt == ref_nt or obs_nt == '.':
      # Impute . as ref_nt
      pass
    else:
      num_edits += 1
  return num_edits

def is_simulated_precise(row, central_col, col_to_ref_nt):
  num_edits = 0
  for col in col_to_ref_nt:
    ref_nt = col_to_ref_nt[col]
    obs_nt = row[col]
    if obs_nt == ref_nt or obs_nt == '.':
      # Impute . as ref_nt
      pass
    else:
      num_edits += 1
  if num_edits == 1:
    if row[central_col] != central_col[0]:
      if row[central_col] != '.':
        return True
  return False

def find_central_col(central_pos, col_nms, substrate):
  best_score = 100
  best_col = None
  for col in col_nms:
    try:
      pos = int(col[1:])
    except ValueError:
      continue
    nt = col[0]
    if nt != substrate: continue
    score = abs(pos - central_pos)
    if score < best_score:
      best_score = score
      best_col = col
  return best_col

##
# Form groups
##
def form_data(exp_nm, start_idx, end_idx):
  '''
    Annotate library design with total count, edited count, fraction edited, etc. 
  '''
  data = _data.load_data(exp_nm, 'ag5a4_profile_subset')
  lib_design, seq_col = _data.get_lib_design(exp_nm)
  lib_nm = _data.get_lib_nm(exp_nm)

  lib_design = lib_design.iloc[start_idx : end_idx + 1]
  ontarget_sites = _data.get_ontarget_sites(lib_design, lib_nm)
  lib_design = lib_design[lib_design['Name (unique)'].isin(ontarget_sites)]

  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  stats_dd = defaultdict(list)
  new_data = dict()

  nms_shared = [nm for nm in nms if nm in data]
  timer = util.Timer(total = len(nms_shared))
  for iter, nm in enumerate(nms_shared):

    df = data[nm]
    seq = nm_to_seq[nm]

    num_mismatches = lambda x, y: sum([bool(n1 != n2) for n1,n2 in zip(x,y)])

    if 'index' in df.columns:
      df = df[[col for col in df.columns if col != 'index']]

    if len(df) == 0: continue


    ## 8/21/19
    '''
      Simulate bystander precision task in 12kChar by using the substrate nucleotide closest to the editor-specific center nt
    '''
    editor = _data.get_editor_nm(exp_nm)
    editor_to_central_pos = {
      'ABE': 6,
      'ABE-CP': 6,
      'AID': 6,
      'BE4': 6,
      'BE4-CP': 8,
      'CDA': 5,
      'eA3A': 6,
      'evoAPOBEC': 5,
    }
    if editor in editor_to_central_pos:
      central_pos = editor_to_central_pos[editor]
    else:
      central_pos = 6

    substrate = 'A' if 'ABE' in editor else 'C'
    nt_cols = [f'{substrate}{pos}' for pos in range(-3, 15) if f'{substrate}{pos}' in df.columns]
    central_col = find_central_col(central_pos, nt_cols, substrate)
    if central_col is None: continue

    mut_cols = [col for col in df.columns if col != 'Count']
    col_to_ref_nt = {col: col[0] for col in mut_cols}
    df_dd = defaultdict(list)
    for idx, row in df.iterrows():
      df_dd['Num. edits'].append(get_num_edits(row, col_to_ref_nt))
      df_dd['Simulated precise'].append(is_simulated_precise(row, central_col, col_to_ref_nt))
    for col in df_dd:
      df[col] = df_dd[col]

    numer = sum(df[df['Simulated precise'] == True]['Count'])
    denom = sum(df[df['Num. edits'] > 0]['Count'])
    sim_precision = numer / denom if denom > 0 else np.nan
    stats_dd['Simulated bystander precision at editor-specific central nt'].append(sim_precision)

    stats_dd['Simulated bystander position'].append(int(central_col[1:]))
    stats_dd['Simulated bystander position, distance to center'].append(int(central_col[1:]) - central_pos)

    edited_ct = sum(df[df['Num. edits'] > 0]['Count'])
    stats_dd['Edited count'].append(edited_ct)

    stats_dd['Name (unique)'].append(nm)

    timer.update()


  stats_df_collected = pd.DataFrame(stats_dd)

  stats_df = lib_design.merge(
    stats_df_collected, 
    on = 'Name (unique)', 
    how = 'outer',
  )

  stats_df.to_csv(out_dir + '%s_%s_%s_stats.csv' % (exp_nm, start_idx, end_idx))
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
    lib_nm = _data.get_lib_nm(treat_nm)

    if lib_nm != '12kChar': continue
    if 'U2OS' in treat_nm: continue

    num_targets = 12000
    num_targets_per_split = 2000

    for start_idx in range(0, num_targets, num_targets_per_split):
      end_idx = start_idx + num_targets_per_split - 1

      # Skip completed
      out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx)
      if os.path.isfile(out_pkl_fn):
        if os.path.getsize(out_pkl_fn) > 0:
          continue

      command = 'python %s.py %s %s %s' % (NAME, treat_nm, start_idx, end_idx)
      script_id = NAME.split('_')[0]

      try:
        mb_file_size = _data.check_file_size(treat_nm, 'ag5a4_profile_subset')
      except FileNotFoundError:
        mb_file_size = 0
      ram_gb = 2
      if mb_file_size > 140:
        ram_gb = 4
      if mb_file_size > 400:
        ram_gb = 8
      if mb_file_size > 1000:
        ram_gb = 16

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, start_idx)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      num_scripts += 1

      # Write qsub commands
      qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=%sG -wd %s %s &' % (ram_gb, _config.SRC_DIR, sh_fn))

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
  form_data(exp_nm, int(start_idx), int(end_idx))

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1], start_idx = sys.argv[2], end_idx = sys.argv[3])
  else:
    gen_qsubs()
    # main()