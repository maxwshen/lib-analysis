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
inp_dir = _config.OUT_PLACE + 'indel_anyindel_pos_c1bpfq_v2/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)


def gather(editor):
  print(editor)
  conds = exp_design_df[exp_design_df['Editor'] == editor]['Name']

  # mdf
  mdf = pd.DataFrame()
  timer = util.Timer(total = len(conds))
  for cond in conds:
    try:
      df = pd.read_csv(inp_dir + f'{cond}.csv', index_col = 0)
      mdf = mdf.append(df, ignore_index = True, sort = False)
    except:
      continue
    timer.update()
  mdf.to_csv(out_dir + f'{editor}.csv')

  # len melt
  mdf = pd.DataFrame()
  timer = util.Timer(total = len(conds))
  for cond in conds:
    try:
      df = pd.read_csv(inp_dir + f'{cond}_len_melt.csv', index_col = 0)
      df['Condition'] = cond
      mdf = mdf.append(df, ignore_index = True, sort = False)
    except:
      continue
    timer.update()
  mdf.to_csv(out_dir + f'{editor}_len_melt.csv')

  # pos_melt by nt
  # Reduce to mean
  indel_lens = range(-20, 15 + 1)
  timer = util.Timer(total = len(indel_lens))
  for indel_len in indel_lens:
    mdf = pd.DataFrame()
    for cond in conds:
      try:
        df = pd.read_csv(inp_dir + f'{cond}_pos_{indel_len}nt.csv', index_col = 0)
        df['Condition'] = cond
        df['Indel length'] = indel_len
        mdf = mdf.append(df, ignore_index = True, sort = False)
      except:
        continue
    mdf.to_csv(out_dir + f'{editor}_pos_{indel_len}nt.csv')
    timer.update()

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

  editors = sorted(list(set(exp_design_df['Editor'])))

  num_scripts = 0
  for editor in editors:
    if editor in ['Cas9', 'UT', 'A3A']:
      continue

    command = 'python %s.py %s' % (NAME, editor)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, editor)
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
def main(argv):
  print(NAME)
  
  editor_nm = argv[0]

  if editor_nm == 'run_all':
    for editor_nm in sorted(list(set(exp_design_df['Editor']))):
      if editor_nm in ['Cas9', 'UT', 'A3A']:
        continue
      print(editor_nm)
      gather(editor_nm)
  else:
    gather(editor_nm)
  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
    # main()