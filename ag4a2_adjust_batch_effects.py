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
inp_dir = _config.OUT_PLACE + 'ag4a_calc_stats/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
batches = exp_design['Batch covariate']
editors = exp_design['Editor']
exp_nm_to_batch = {exp_nm: batch for exp_nm, batch in zip(exp_nms, batches)}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

##
# Filters
##
def filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm):
  # timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    pos_idx = _data.pos_to_idx(row['Position'], treat_nm)
    kdx = nt_to_idx[row['Obs nt']]
    ref_nt = row['Ref nt']
    
    for nm in adj_d:
      seq = nm_to_seq[nm]
      if seq[pos_idx] == ref_nt:
        t = adj_d[nm]
        t[pos_idx][kdx] = 0
        adj_d[nm] = t
    # timer.update()

  return adj_d

##
# Support
##
def get_combined_grid_profiles(adf, groups, group_to_nms):
  group_dfs = dict()

  print('Combining stats across groups (batches, editors)...')

  timer = util.Timer(total = len(groups))
  for group_nm in groups:
    nms = group_to_nms[group_nm]
    
    mdf = pd.DataFrame()
    for nm in nms:
      df = adf[nm]
      mdf = mdf.append(df, ignore_index = True)

    mdf = mdf.groupby(list(mdf.columns)).size().reset_index().rename(columns = {0: 'Pool count'})
    mdf = mdf.sort_values(by = 'Pool count', ascending = False).reset_index()
    mdf['Pool fraction'] = mdf['Pool count'] / len(nms)
    group_dfs[group_nm] = mdf

    timer.update()

  return group_dfs

##
# Main iterator
##
def adjust_batch_effects():
  # Gather statistics
  be_treatments = [s for s in treat_control_df['Treatment'] if 'Cas9' not in s]
  mdf = pd.DataFrame()
  timer = util.Timer(total = len(be_treatments))
  print('Loading stats from each condition...')
  for treat_nm in be_treatments:
    df = pd.read_csv(inp_dir + '%s.csv' % (treat_nm), index_col = 0)
    df['Treatment'] = treat_nm
    df['Batch'] = exp_nm_to_batch[treat_nm]
    df['Editor'] = exp_nm_to_editor[treat_nm]
    mdf = mdf.append(df, ignore_index = True)
    timer.update()

  mdf['Log mean activity'] = np.log10(mdf['Mean activity'])

  # ANOVA calculations
  from scipy.stats import f_oneway

  print('Calculating ANOVA on each position+mutation combination to identify batch effects...')
  dd = defaultdict(list)
  set_pos = set(mdf['Position'])
  timer = util.Timer(total = len(set_pos))
  for pos in set_pos:
    timer.update()
    for ref_nt in set(mdf['Ref nt']):
      for obs_nt in set(mdf['Obs nt']):
        crit = (mdf['Position'] == pos) & \
               (mdf['Ref nt'] == ref_nt) & \
               (mdf['Obs nt'] == obs_nt)
        if len(mdf[crit]) == 0:
            continue
        if pos in [22, 23]:
            continue
        args = tuple([mdf[crit & (mdf['Batch'] == batch_nm)]['Mean activity'] for batch_nm in set(mdf[crit]['Batch'])])
        fstat, pval = f_oneway(*args)
        dd['pval'].append(pval)
        dd['Statistic'].append(fstat)
        dd['Position'].append(pos)
        dd['Ref nt'].append(ref_nt)
        dd['Obs nt'].append(obs_nt)

  stats_df = pd.DataFrame(dd)
  stats_df['-log10p'] = -np.log10(stats_df['pval'])

  # Apply Bonferroni p-value cutoff
  print('Finding significant batch effects with a Bonferroni corrected p-value threshold...')
  pval = 0.005
  bonf_threshold = pval / len(stats_df)
  stats_df['bonferroni accept'] = (stats_df['pval'] <= bonf_threshold)
  stats_df.to_csv(out_dir + 'mutation_dec.csv')


  '''
    Identify mutations for removal
    At mutations passing Bonferroni corrected ANOVA test,
    identify batches where mutations are frequent
  '''
  print('Identifying batches to remove mutations from...')
  to_remove = stats_df[stats_df['bonferroni accept'] == True]
  dd = defaultdict(list)
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    timer.update()
    pos = row['Position']
    ref_nt = row['Ref nt']
    obs_nt = row['Obs nt']
    crit = (mdf['Position'] == pos) & \
           (mdf['Ref nt'] == ref_nt) & \
           (mdf['Obs nt'] == obs_nt)
    means = {
      batch_nm: np.mean(mdf[crit & (mdf['Batch'] == batch_nm)]['Mean activity']) \
      for batch_nm in set(mdf[crit]['Batch'])
    }
    mean_vals = list(means.values())
    mean_means = np.mean(mean_vals)
    if max(mean_vals) - min(mean_vals) < 0.002:
      continue
    for batch_nm in means:
      if means[batch_nm] >= mean_means or means[batch_nm] >= 0.005:
        dd['Position'].append(pos)
        dd['Ref nt'].append(ref_nt)
        dd['Obs nt'].append(obs_nt)
        dd['Batch'].append(batch_nm)
  batch_muts_to_remove = pd.DataFrame(dd)
  batch_muts_to_remove.to_csv(out_dir + 'removed_batch_effects.csv')

  # Remove mutations
  print('Removing batch effects in each condition...')
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    adj_d = _data.load_data(treat_nm, 'ag4_poswise_be_adjust')
    lib_design, seq_col = _data.get_lib_design(treat_nm)
    nms = lib_design['Name (unique)']
    seqs = lib_design[seq_col]
    nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

    batch = exp_nm_to_batch[treat_nm]
    to_remove = batch_muts_to_remove[batch_muts_to_remove['Batch'] == batch]
    adj_d = filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm)

    with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
      pickle.dump(adj_d, f)

    timer.update()

  return

##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Function calls
  adjust_batch_effects()

  return


if __name__ == '__main__':
  main()
