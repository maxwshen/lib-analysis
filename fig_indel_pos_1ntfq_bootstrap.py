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
inp_dir = _config.OUT_PLACE + 'indel_anyindel_pos_c1bpfq_v2/'
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
def get_statistics(cond):

  df1 = pd.read_csv(inp_dir + f'{cond}_pos_1nt.csv', index_col = 0)
  df2 = pd.read_csv(inp_dir + f'{cond}_pos_-1nt.csv', index_col = 0)
  mdf = df1.append(df2, ignore_index = True)

  bs_dd = defaultdict(list)
  
  positions = [col for col in mdf.columns if col != 'Name']
  timer = util.Timer(total = len(positions))
  for pos in positions:
    
    dfs = mdf[pos]

    means = []
    for bs_idx in range(1000):
      bs_data = np.random.choice(dfs, size = len(dfs), replace = True)
      means.append(np.mean(bs_data))

    bs_dd['Position'].append(pos)
    bs_dd['Mean'].append(np.mean(dfs))
    bs_dd['Mean - stderr'].append(np.percentile(means, 50 - 34))
    bs_dd['Mean + stderr'].append(np.percentile(means, 50 + 34))
    bs_dd['2.5th percentile'].append(np.percentile(means, 2.5))
    bs_dd['97.5th percentile'].append(np.percentile(means, 97.5))
    timer.update()

  bs_df = pd.DataFrame(bs_dd)
  bs_df = bs_df.sort_values(by = 'Position').reset_index()

  bs_df.to_csv(out_dir + f'{cond}.csv')

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
  all_conds = treatments + controls

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
def main(argv):
  print(NAME)

  cond = argv[0]

  get_statistics(cond)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
    # main()