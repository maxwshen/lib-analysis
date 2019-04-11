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
inp_dir = _config.OUT_PLACE + 'ag4_poswise_be_adjust/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Statistics
##
def ben_hoch_fdr(df, fdr_threshold):
  other_distribution = df[df['pval'] > 0.995]
  df = df[df['pval'] <= 0.995]
  df = df.sort_values(by = 'pval')
  df = df.reset_index(drop = True)

  fdr_decs, hit_reject = [], False
  for idx, pval in enumerate(df['pval']):
    if hit_reject:
      dec = False
    else:
      fdr_critical = ((idx + 1) / len(df)) * fdr_threshold
      dec = bool(pval <= fdr_critical)
    fdr_decs.append(dec)
    if dec is False and hit_reject is True:
      hit_reject = False
  df['FDR accept'] = fdr_decs

  other_distribution['FDR accept'] = False
  df = df.append(other_distribution, ignore_index = True)
  return df

##
# Filters
##
def filter_freqzero_background_mutations(to_remove, adj_d, nm_to_seq):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    pos_idx = row['Position index']
    kdx = nt_to_idx[row['Obs nt']]
    ref_nt = row['Ref nt']
    
    for nm in adj_d:
      seq = nm_to_seq[nm]
      if seq[pos_idx] == ref_nt:
        t = adj_d[nm]
        t[pos_idx][kdx] = 0
        adj_d[nm] = t
    timer.update()

  return adj_d

##
# Main iterator
##
def adjust_treatment_control(treat_nm):
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''

  adj_d = _data.load_data(treat_nm, 'ag4_poswise_be_adjust')
  # adj_d = pickle.load(open(inp_dir + '%s.pkl' % (treat_nm), 'rb'))

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  '''
    Filter treatment mutations that match the unedited background profile
    using the statistic: fraction of target sites with non-zero event frequency
  '''
  print('Gathering statistics on treatment mutations matching background profile by frequency of zeros...')
  dd = defaultdict(list)
  timer = util.Timer(total = len(adj_d))
  for nm in adj_d:
    timer.update()
    pw = adj_d[nm]
    seq = nm_to_seq[nm]
    for jdx in range(len(pw)):
      tot = sum(pw[jdx])
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
  df = df[df['Total count'] >= 50]

  # Form stats_df and find p for binomial, which is typically ~0.99
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

        dd['Num target sites with zero'].append(num_zeros)
        dd['Total num target sites'].append(total)
        dd['Frequency of zero in target sites'].append(num_zeros / total)
        dd['Mean activity'].append(np.mean(dfs_freq))
        dd['Position index'].append(pos_idx)
        dd['Position'].append(_data.idx_to_pos(pos_idx, treat_nm))
        dd['Obs nt'].append(obs_nt)
        dd['Ref nt'].append(ref_nt)

  fz_df = pd.DataFrame(dd)

  baseline_pos_range = pos_range[-5:]
  max_mean_activity = 0.025
  min_num_targets = 50

  crit = (fz_df['Position index'].isin(baseline_pos_range)) & \
         (fz_df['Mean activity'] <= max_mean_activity) & \
         (fz_df['Total num target sites'] >= min_num_targets)
  bg_bin_p = np.mean(fz_df[crit]['Frequency of zero in target sites'])
  if np.isnan(bg_bin_p):
    raise ValueError

  pvals = []
  timer = util.Timer(total = len(fz_df))
  for idx, row in fz_df.iterrows():
    total = row['Total num target sites']
    numzero = row['Num target sites with zero']
    pval = binom.cdf(numzero, total, bg_bin_p)
    pvals.append(pval)
    timer.update()
  fz_df['pval'] = pvals

  fz_fdr_threshold = 0.001
  fz_df = ben_hoch_fdr(fz_df, fz_fdr_threshold)
  fz_df.to_csv(out_dir + '%s_fraczero_dec.csv' % (treat_nm))

  print('Filtering treatment mutations matching background profile by frequency of zeros...')
  to_remove = fz_df[fz_df['FDR accept'] == False]
  adj_d = filter_freqzero_background_mutations(to_remove, adj_d, nm_to_seq)


  ##
  # Write
  ##
  with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
    pickle.dump(adj_d, f)

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

    if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
      continue
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
    qsub_commands.append('qsub -V -l h_rt=2:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
      treat_nm = row['Treatment'], row['Control']
      if 'Cas9' in treat_nm:
        continue
      print(treat_nm)
      if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
        print('... already complete')
      else:
        adjust_treatment_control(treat_nm)

  else:
    adjust_treatment_control(treat_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()