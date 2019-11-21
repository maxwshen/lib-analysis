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
inp_dir = _config.OUT_PLACE + 'indel_anyindel_pos/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)


def indel_anyindel_seq(exp_nm):
  '''
    Annotate indels with related sequence context (e.g., bases in deletions) 
  '''
  df = pd.read_csv(inp_dir + '%s.csv' % (exp_nm), index_col = 0)

  lib_design, seq_col = _data.get_lib_design(exp_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  indel_dd = defaultdict(list)
  all_nms = set(df['Name'])
  timer = util.Timer(total = len(df))
  for idx, row in df.iterrows():
    # dfs = df[df['Name'] == nm]

    nm = row['Name']
    seq = nm_to_seq[nm]
    left_del_nt = np.nan
    right_del_nt = np.nan
    del_nts = np.nan

    if row['Category'] == 'del':
      start_pos = int(row['Indel start adj'])
      start_idx = _data.pos_to_idx(start_pos, exp_nm)

      end_pos = int(row['Indel end adj'])
      end_idx = _data.pos_to_idx(end_pos, exp_nm)

      if start_idx >= 0 and end_idx <= len(seq):
        del_nts = seq[start_idx : end_idx]
        left_del_nt = del_nts[0]
        right_del_nt = del_nts[-1]

    indel_dd['Left del nt'].append(left_del_nt)
    indel_dd['Right del nt'].append(right_del_nt)
    indel_dd['Del nts'].append(del_nts)

    timer.update()

  for col in indel_dd:
    df[col] = indel_dd[col]

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

  num_scripts = 0
  for idx, row in treat_control_df.iterrows():
    treat_nm = row['Treatment']

    command = 'python %s.py %s' % (NAME, treat_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, treat_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  
  indel_anyindel_seq(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()