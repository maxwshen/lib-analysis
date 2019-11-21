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
    Investigate if 1 nt deletions at abasic site are related to microhomology
    Control for position by focusing only on pos 5
  '''
  df = pd.read_csv(inp_dir + '%s.csv' % (exp_nm), index_col = 0)

  lib_design, seq_col = _data.get_lib_design(exp_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  dd = defaultdict(list)
  all_nms = set(df['Name'])

  five_idx = _data.pos_to_idx(4, exp_nm)

  import code; code.interact(local=dict(globals(), **locals()))

  timer = util.Timer(total = len(all_nms))
  for nm in all_nms:
    dfs = df[df['Name'] == nm]

    seq = nm_to_seq[nm]
    accept = bool(seq[five_idx] == 'C') & (seq[five_idx + 1] != 'C')

    for jdx in range(five_idx - 1, -1, -1):
      if seq[jdx] != 'C':
        break
    num_c = abs(five_idx - jdx)


    if not accept:
      continue

    crit = (dfs['Category'] == 'del') & (dfs['Indel length'] == 1) & (dfs['Indel end adj'] == 5.0)
    row = dfs[crit]

    if len(row) == 0:
      dd['Frequency'].append(0)
    else:
      dd['Frequency'].append(sum(row['Frequency']))

    dd['Num C'].append(num_c)
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

  num_scripts = 0
  for idx, row in treat_control_df.iterrows():
    treat_nm = row['Treatment']
    if 'ABE' in treat_nm or 'Cas9' in treat_nm:
      continue

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