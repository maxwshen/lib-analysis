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
inp_dir = _config.OUT_PLACE + 'indel_anyindel_pos/'
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
  import matplotlib
  matplotlib.use('Pdf')
  import matplotlib.pyplot as plt
  import seaborn as sns

  # Position indel heatmap
  ddf = dict()
  for indel_len in range(-20, 15+1):
    df = pd.read_csv(inp_dir + '%s_pos_%snt.csv' % (nm, indel_len), index_col = 0)
    ddf[indel_len] = df

  hm = pd.DataFrame()
  dd = defaultdict(list)
  for indel_len in ddf:
    if indel_len == 0:
      continue
    df = ddf[indel_len]
    means = df.drop(columns = 'Name').apply(np.mean)
    hm = hm.append(means, ignore_index = True)
    dd['Indel length'].append(indel_len)
  hm.index = dd['Indel length']

  filt_cols = [str(s) for s in range(-10, 30)]
  hm = hm[filt_cols]

  sns.clustermap(
      data = hm,
      row_cluster = False,
      col_cluster = False,
  #     standard_scale = 0,
      figsize = (13, 13),
  )
  plt.xlabel('Position')
  plt.ylabel('Indel length')
  plt.title(nm)
  plt.savefig(out_dir + '%s_heatmappos.pdf' % (nm))
  plt.savefig(out_dir + '%s_heatmappos.jpg' % (nm))
  plt.close()

  # Insertions from template slipping
  df = pd.read_csv(inp_dir + '%s.csv' % (nm), index_col = 0)
  crit = (df['Category'] == 'ins')
  dfs = df[crit]
  dfs['Insertion length'] = dfs['Indel length']

  sns.jointplot(
      x = 'Insertion length',
      y = 'MH length',
      data = dfs,
      kind = 'scatter',
      joint_kws = dict(
          alpha = len(dfs)**(-1/2),
      ),
  )

  from scipy.stats import pearsonr
  pr = pearsonr(dfs['MH length'], dfs['Insertion length'])
  plt.title('%s\nN = %s\n%s' % (nm, len(dfs), pr))
  plt.tight_layout()
  plt.savefig(out_dir + '%s_insmh.pdf' % (nm))
  plt.savefig(out_dir + '%s_insmh.jpg' % (nm))
  plt.close()

  # Mean bar

  df = pd.read_csv(inp_dir + '%s_pos_melt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Position',
      y = 'Frequency',
      data = df,
      ax = ax,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_poswiseindel.pdf' % (nm))
  plt.savefig(out_dir + '%s_poswiseindel.jpg' % (nm))
  plt.close()

  df = pd.read_csv(inp_dir + '%s_pos_melt_-1nt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Position',
      y = 'Frequency',
      data = df,
      ax = ax,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_poswise1ntindel.pdf' % (nm))
  plt.savefig(out_dir + '%s_poswise1ntindel.jpg' % (nm))
  plt.close()

  df = pd.read_csv(inp_dir + '%s_len_melt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Indel length',
      y = 'Frequency',
      data = df,
      ax = ax,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_lengthfreq.pdf' % (nm))
  plt.savefig(out_dir + '%s_lengthfreq.jpg' % (nm))
  plt.close()

  # Median bar

  df = pd.read_csv(inp_dir + '%s_pos_melt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Position',
      y = 'Frequency',
      data = df,
      ax = ax,
      estimator = np.median,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_medianpwi.pdf' % (nm))
  plt.savefig(out_dir + '%s_medianpwi.jpg' % (nm))
  plt.close()

  df = pd.read_csv(inp_dir + '%s_pos_melt_-1nt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Position',
      y = 'Frequency',
      data = df,
      ax = ax,
      estimator = np.median,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_medianpwi1nt.pdf' % (nm))
  plt.savefig(out_dir + '%s_medianpwi1nt.jpg' % (nm))
  plt.close()

  df = pd.read_csv(inp_dir + '%s_len_melt.csv' % (nm), index_col = 0)
  fig, ax = plt.subplots(figsize = (12, 6))
  sns.barplot(
      x = 'Indel length',
      y = 'Frequency',
      data = df,
      ax = ax,
      estimator = np.median,
  )
  plt.title(nm)
  plt.xticks(rotation = 90);
  plt.savefig(out_dir + '%s_medianlenfreq.pdf' % (nm))
  plt.savefig(out_dir + '%s_medianlenfreq.jpg' % (nm))
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

  treatments = list(set(treat_control_df['Treatment']))
  controls = list(set(treat_control_df['Control']))
  # all_conds = treatments + controls
  all_conds = controls

  num_scripts = 0
  for cond in all_conds:

    out_fn = out_dir + '%s_medianlenfreq.jpg' % (cond)
    if os.path.isfile(out_fn):
      continue

    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, cond)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=2:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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