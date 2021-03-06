# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
from scipy.stats import binom

# Default params
inp_dir = _config.OUT_PLACE + 'ag4b4_filter_inprofile_batch_effects/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Statistics
##

##
# Main iterator
##
def fig_editing_profiles(treat_nm):
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''

  adj_d = pickle.load(open(inp_dir + '%s.pkl' % (treat_nm), 'rb'))


  lib_design, seq_col = _data.get_lib_design(treat_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  lib_nm = _data.get_lib_nm(treat_nm)
  ontarget_nms = set(_data.get_ontarget_sites(lib_design, lib_nm))

  '''
    Filter treatment mutations that match the unedited background profile
    using the statistic: fraction of target sites with non-zero event frequency
  '''
  print('Forming long df...')
  dd = defaultdict(list)
  timer = util.Timer(total = len(adj_d))
  for nm in adj_d:
    timer.update()

    if nm not in ontarget_nms:
      continue

    pw = adj_d[nm]
    seq = nm_to_seq[nm]
    for jdx in range(len(pw)):
      tot = np.nansum(pw[jdx])
      ref_nt = seq[jdx]
      ref_idx = nt_to_idx[ref_nt]
      for kdx in range(len(pw[jdx])):
        if kdx == ref_idx:
          continue

        count = pw[jdx][kdx]
        dd['Count'].append(count)
        dd['Total count'].append(tot)
        dd['Obs nt'].append(nts[kdx])
        dd['Ref nt'].append(ref_nt)
        if tot == 0:
          dd['Frequency'].append(np.nan)
        else:
          dd['Frequency'].append(count / tot)
        dd['Position index'].append(jdx)
        dd['Position'].append(_data.idx_to_pos(jdx, treat_nm))
        dd['Name'].append(nm)

  df = pd.DataFrame(dd)
  df = df[df['Total count'] >= 100]
  n_targetsites_in_condition = len(df)

  # Form stats_df
  dd = defaultdict(list)
  pos_range = sorted(set(df['Position index']))
  timer = util.Timer(total = len(pos_range))
  for pos_idx in pos_range:
    timer.update()
    df_s1 = df[df['Position index'] == pos_idx]
    for ref_nt in nts:
      df_s2 = df_s1[df_s1['Ref nt'] == ref_nt]
      for obs_nt in nts:
        if obs_nt == ref_nt:
          continue

        crit = (df_s2['Obs nt'] == obs_nt)
        dfs = df_s2[crit]
        dfs_freq = dfs['Frequency']

        num_zeros = sum(dfs_freq == 0)
        total = len(dfs_freq)
        if total == 0:
          continue

        dd['Num target sites with zero for mutation'].append(num_zeros)
        dd['Total num target sites for mutation'].append(total)
        dd['Frequency of zero in target sites for mutation'].append(num_zeros / total)
        dd['Num target sites in condition'].append(n_targetsites_in_condition)
        dd['Mean activity'].append(np.mean(dfs_freq))
        dd['Position index'].append(pos_idx)
        dd['Position'].append(_data.idx_to_pos(pos_idx, treat_nm))
        dd['Obs nt'].append(obs_nt)
        dd['Ref nt'].append(ref_nt)

  hm_df = pd.DataFrame(dd)
  hm_df.to_csv(out_dir + '%s.csv' % (treat_nm))

  # Median normalize
  background_range = range(25, 34 + 1)

  for ref_nt in nts:
    for obs_nt in nts:
      if obs_nt == ref_nt:
        continue

      crit = (hm_df['Ref nt'] == ref_nt) & (hm_df['Obs nt'] == obs_nt) & (~np.isnan(hm_df['Mean activity']))
      medi = np.nanmedian(hm_df[crit & (hm_df['Position'].isin(background_range))]['Mean activity'])
      hm_df.loc[crit, 'Mean activity'] = hm_df.loc[crit, 'Mean activity'].apply(lambda x: max(0, x - medi))

  hm_df.to_csv(out_dir + '%s_median_bg_adj.csv' % (treat_nm))

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

    if os.path.exists(out_dir + '%s.csv' % (treat_nm)):
      continue
    if 'Cas9' in treat_nm:
      continue
    if '12kChar' not in treat_nm:
      continue

    command = 'python %s.py %s' % (NAME, treat_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, treat_nm)
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
def main(treat_nm = ''):
  print(NAME)
  
  # Function calls
  if treat_nm == '':

    treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

    for idx, row in treat_control_df.iterrows():
      treat_nm = row['Treatment']
      if '12kChar' not in treat_nm:
        continue
      if 'Cas9' in treat_nm:
        continue
      print(treat_nm)
      # fig_editing_profiles(treat_nm)

      if os.path.exists(out_dir + '%s.csv' % (treat_nm)):
        print('... already complete')
      else:
        fig_editing_profiles(treat_nm)
      

  else:
    fig_editing_profiles(treat_nm)

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(treat_nm = sys.argv[1])
  # else:
  #   gen_qsubs()
  main()