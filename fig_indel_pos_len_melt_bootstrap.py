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
inp_dir = _config.OUT_PLACE + 'indel_anyindel_pos_c1bpfq_editor/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)


##
# Form groups
##
def get_statistics(editor):

  df = pd.read_csv(inp_dir + f'{editor}_len_melt.csv', index_col = 0)

  bs_dd = defaultdict(list)
  
  indel_lens = set(df['Indel length'])
  timer = util.Timer(total = len(indel_lens))
  for indel_len in indel_lens:
    dfs = df[df['Indel length'] == indel_len]
    
    means = []
    for bs_idx in range(1000):
      bs_data = np.random.choice(dfs['Frequency'], size = len(dfs), replace = True)
      means.append(np.mean(bs_data))

    bs_dd['Indel length'].append(indel_len)
    bs_dd['Mean'].append(np.mean(dfs['Frequency']))
    bs_dd['Mean - stderr'].append(np.percentile(means, 50 - 34))
    bs_dd['Mean + stderr'].append(np.percentile(means, 50 + 34))
    bs_dd['2.5th percentile'].append(np.percentile(means, 2.5))
    bs_dd['97.5th percentile'].append(np.percentile(means, 97.5))
    timer.update()

  bs_df = pd.DataFrame(bs_dd)
  bs_df = bs_df.sort_values(by = 'Indel length').reset_index()

  bs_df.to_csv(out_dir + f'{editor}.csv')

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

  main_editors = [
    'ABE',
    'ABE-CP1040',
    'BE4',
    'AID',
    'CDA',
    'evoAPOBEC',
    'eA3A',
    'BE4-CP1028',
  ]

  num_scripts = 0
  for editor in main_editors:

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

  editor = argv[0]

  get_statistics(editor)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
    # main()