# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/home/unix/maxwshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

lib_design = pd.read_csv(_config.DATA_DIR + 'library_12kChar.csv')


##
# Functions
##
def get_match_count(grna, seq):
  expected_guide_start_12kchar = 22
  seq_g = seq[expected_guide_start_12kchar : expected_guide_start_12kchar + 20]
  mm = 0
  for nt1, nt2 in zip(grna, seq_g):
    if nt1 == nt2:
      mm += 1
  return mm

##
# Form data
##
def form_data(exp_nm):
  data = _data.load_data(exp_nm, 'ae_newgenotype_Cas9_adjust')
  lib_design, seq_col = _data.get_lib_design(exp_nm)

  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  grnas = lib_design['gRNA (20nt)']
  design_cats = lib_design['Design category']
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}
  nm_to_grna = {nm: grna for nm, grna in zip(nms, grnas)}
  nm_to_design_cat = {nm: design_cat for nm, design_cat in zip(nms, design_cats)}


  signal_cats = ['wildtype', 'del', 'ins', 'del_notatcut', 'ins_notatcut', 'combination_indel']
  edited_cats = ['del', 'ins', 'del_notatcut', 'ins_notatcut', 'combination_indel']

  dd = defaultdict(list)

  timer = util.Timer(total = len(data))
  for nm in data:
    d = data[nm]
    seq = nm_to_seq[nm]
    grna = nm_to_grna[nm]
    design_cat = nm_to_design_cat[nm]

    signal_count = sum(d[d['Category'].isin(signal_cats)]['Count'])
    edit_count = sum(d[d['Category'].isin(edited_cats)]['Count'])

    if signal_count != 0:
      edited_frac = edit_count / signal_count
    else:
      edited_frac = np.nan

    dd['Edited fraction'].append(edited_frac)
    dd['Edited count'].append(edit_count)
    dd['Unedited count'].append(signal_count - edit_count)
    dd['Total count'].append(signal_count)

    match_count = get_match_count(grna, seq)

    if design_cat == 'guideseq':
      category = 'Off-target series'
      subcategory = nm.split('_')[2] # gene name
    elif design_cat == 'mismatch':
      category = 'Mismatch series'
      subcategory = nm.split('_')[1] # series number
    elif design_cat == 'chipseq':
      category = 'Chip series'
    elif design_cat == 'vivo':
      category = 'vivo'
      subcategory = 'vivo'
    else:
      assert match_count == 20, 'fail'
      category = 'On-target'
      subcategory = 'On-target'

    dd['Match count'].append(int(match_count))
    dd['Category'].append(category)
    dd['Subcategory'].append(subcategory)
    dd['Name'].append(nm)

    timer.update()


  df = pd.DataFrame(dd)
  df.to_csv(out_dir + '%s.csv' % (exp_nm))

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
    if 'Cas9' not in treat_nm:
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


@util.time_dec
def main(exp_nm = ''):
  print(NAME)

  # Function calls
  form_data(exp_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()