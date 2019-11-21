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

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Form groups
##
def form_data(exp_nm):
  data = _data.load_data(exp_nm, 'ag4b4_filter_inprofile_batch_effects')
  lib_design, seq_col = _data.get_g4_lib_design(exp_nm)

  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''

  dd = defaultdict(list)

  timer = util.Timer(total = len(data))
  for nm in data:
    pw = data[nm]
    seq = nm_to_seq[nm]

    for jdx in range(len(pw)):
      pos = _data.idx_to_pos(jdx, exp_nm)
      ref_nt = seq[jdx]
      ref_idx = nt_to_idx[ref_nt]
      total = sum(pw[jdx])
      for kdx in range(len(pw[jdx])):
        if kdx == ref_idx:
          continue

        count = pw[jdx][kdx]

        dd['Count'].append(count)
        dd['Total count'].append(total)
        dd['Obs nt'].append(nts[kdx])
        dd['Ref nt'].append(ref_nt)
        dd['Position'].append(pos)
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
    if 'Cas9' in treat_nm:
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
  
  # Function calls
  form_data(exp_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()