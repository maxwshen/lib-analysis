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
inp_dir = _config.OUT_PLACE + 'c_poswise_basic/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Form groups
##
def plot_data(nm):
  import matplotlib.pyplot as plt
  import seaborn as sns

  df = pd.read_csv(inp_dir + '%s.csv' % (nm), index_col = 0)
  df['Frequency'] = df['Count'] / df['Total count']
  
  # Readcount distribution
  dfs = df[df['Total count'] > 0]
  dfs['Log10 total count'] = np.log10(dfs['Total count'])
  sns.distplot(dfs['Log10 total count'])
  plt.title(nm)
  plt.savefig(out_dir + '%s_readcount.pdf' % (nm))
  plt.savefig(out_dir + '%s_readcount.jpg' % (nm))
  plt.close()
  
  palette = {nt: color for nt, color in zip(list('ACGT'), sns.color_palette("hls", 4))}

  dfs = df[df['Total count'] >= 100]
  fig, ax = plt.subplots(figsize = (30, 6))
  sns.barplot(x = 'Position',
              y = 'Frequency',
              data = dfs,
              hue = 'Ref nt',
              palette = palette,
              hue_order = list('ACGT'),
  )
  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_poswise.pdf' % (nm))
  plt.savefig(out_dir + '%s_poswise.jpg' % (nm))
  plt.close()

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
    qsub_commands.append('qsub -V -P regevlab -l h_rt=1:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  plot_data(exp_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()