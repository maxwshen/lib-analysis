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
inp_dir = _config.OUT_PLACE + 'ag4b_old_adjust_bg_nobatch/'
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

  df = pd.read_csv(inp_dir + '%s_fraczero_dec.csv' % (nm), index_col = 0)
  df = df[~df['Position'].isin([21, 22, 23])]

  stats_df = df
  vmin = min(stats_df['Frequency of zero in target sites'])
  vmax = max(stats_df['Frequency of zero in target sites'])
  fig, axes = plt.subplots(1, 4, figsize = (8, 16), sharey = True)
  for ref_nt, ax in zip(list('ACGT'), axes):
    stats_df_filt = stats_df[stats_df['Ref nt'] == ref_nt]
    stats_df_filt = stats_df_filt[~stats_df_filt['Position'].isin([22, 23])]
    pv = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'Frequency of zero in target sites')
    sns.heatmap(pv, annot = True, cbar = False, ax = ax, vmin = vmin, vmax = vmax)
    ax.set_title(ref_nt)

  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_fraczero.pdf' % (nm))
  plt.savefig(out_dir + '%s_fraczero.jpg' % (nm))
  plt.close()


  stats_df = df
  vmin = min(stats_df['Mean activity'])
  vmax = max(stats_df['Mean activity'])
  fig, axes = plt.subplots(1, 4, figsize = (12, 16), sharey = True)
  for ref_nt, ax in zip(list('ACGT'), axes):
    stats_df_filt = stats_df[stats_df['Ref nt'] == ref_nt]
    stats_df_filt = stats_df_filt[~stats_df_filt['Position'].isin([22, 23])]
    pv = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'Mean activity')
    sns.heatmap(pv, annot = True, cbar = False, ax = ax, fmt = '.1E', vmin = vmin, vmax = vmax, cmap = sns.cm.rocket_r)
    ax.set_title(ref_nt)

  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_meanedit.pdf' % (nm))
  plt.savefig(out_dir + '%s_meanedit.jpg' % (nm))
  plt.close()


  # Repeat, but with FDR filtering

  stats_df = df
  vmin = min(stats_df['Frequency of zero in target sites'])
  vmax = max(stats_df['Frequency of zero in target sites'])
  fig, axes = plt.subplots(1, 4, figsize = (8, 16), sharey = True)
  for ref_nt, ax in zip(list('ACGT'), axes):
    stats_df_filt = stats_df[stats_df['Ref nt'] == ref_nt]
    stats_df_filt = stats_df_filt[~stats_df_filt['Position'].isin([22, 23])]
    pv = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'Frequency of zero in target sites')
    mask = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'FDR accept')
    mask = (mask == False)
    sns.heatmap(pv, annot = True, cbar = False, ax = ax, vmin = vmin, vmax = vmax, mask = mask)
    ax.set_title(ref_nt)

  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_fraczerofdr.pdf' % (nm))
  plt.savefig(out_dir + '%s_fraczerofdr.jpg' % (nm))
  plt.close()


  stats_df = df
  vmin = min(stats_df['Mean activity'])
  vmax = max(stats_df['Mean activity'])
  fig, axes = plt.subplots(1, 4, figsize = (12, 16), sharey = True)
  for ref_nt, ax in zip(list('ACGT'), axes):
    stats_df_filt = stats_df[stats_df['Ref nt'] == ref_nt]
    stats_df_filt = stats_df_filt[~stats_df_filt['Position'].isin([22, 23])]
    pv = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'Mean activity')
    mask = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'FDR accept')  
    mask = (mask == False)
    sns.heatmap(pv, annot = True, cbar = False, ax = ax, fmt = '.1E', vmin = vmin, vmax = vmax, mask = mask, cmap = sns.cm.rocket_r)
    ax.set_title(ref_nt)

  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_meaneditfdr.pdf' % (nm))
  plt.savefig(out_dir + '%s_meaneditfdr.jpg' % (nm))
  plt.close()


  # Normalized

  stats_df = df[~df['Position'].isin([21, 22, 23])]
  stats_df['Mean activity'] = 100 * stats_df['Mean activity'] / np.max(stats_df['Mean activity'])
  fig, axes = plt.subplots(1, 4, figsize = (12, 16), sharey = True)
  for ref_nt, ax in zip(list('ACGT'), axes):
    stats_df_filt = stats_df[stats_df['Ref nt'] == ref_nt]
    pv = stats_df_filt.pivot(index = 'Position', columns = 'Obs nt', values = 'Mean activity')
    sns.heatmap(
      pv, 
      annot = True, 
      cbar = False, 
      ax = ax, 
      fmt = '.2f',
      cmap = sns.cm.rocket_r
    )
    ax.set_title(ref_nt)

  plt.title('%s' % (nm))
  plt.savefig(out_dir + '%s_normmeanedit.pdf' % (nm))
  plt.savefig(out_dir + '%s_normmeanedit.jpg' % (nm))
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
    qsub_commands.append('qsub -V -l h_rt=1:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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