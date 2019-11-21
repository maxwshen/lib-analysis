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
inp_dir = _config.OUT_PLACE + 'ag4a2_adjust_batch_effects/'
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

editor_set = set(editors)
editor_set.remove('UT')
editor_set.remove('Cas9')


##
# Statistics
##
def ben_hoch_fdr(df, fdr_threshold):
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

  return df

def get_normalized_rank_matrix(pv):
  normalized_rank_matrix = pv.dropna()
  for condition in normalized_rank_matrix.columns:
    temp = normalized_rank_matrix[condition]
    temp = temp.sort_values(ascending = False)
    temp = temp.reset_index()
    temp = temp.drop(columns = condition)
    temp['Normalized rank'] = (temp.index + 1) / len(temp)
    temp = temp.set_index('Mutation')
    normalized_rank_matrix[condition] = [temp.loc[s]['Normalized rank'] for s in list(normalized_rank_matrix.index)]
  return normalized_rank_matrix

def get_normalized_rank_vector(mutation, pv_nona):
  for treatment in pv_nona.columns:
    col = pv_nona[treatment]
    temp = col.sort_values(by = treatment)
    temp = temp.reset_index()
  return

def robust_rank_aggregation_pval(mutation, normalized_rank_matrix):
  '''
    Note: RRA yields an upper bound on p-values, so values can be > 1
  '''
  from scipy.stats import beta
  normalized_rank_vec = sorted(normalized_rank_matrix.loc[mutation])
  n = len(normalized_rank_vec)
  pvals = []
  for idx, norm_rank in enumerate(normalized_rank_vec):
    pval = 0
    for k in range(idx + 1, n + 1):
      pval += beta.cdf(norm_rank, k, n + 1 - k)
    pvals.append(pval)
#     print(normalized_rank_vec)
#     print(pvals)
  final_pval_upper_bound = min(pvals) * n
  return final_pval_upper_bound

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
def identify_editor_effects():
  # Gather statistics
  print('Loading data...')
  be_treatments = [s for s in treat_control_df['Treatment'] if 'Cas9' not in s]
  mdf = pd.DataFrame()
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    '''
      Uses ag4b's csvs for mean activity after setting batch effects to 0, but
      before filtering background mutations.

      Adjusts adj_d pw matrices from ag4a2_adjust_batch_effects/
    '''    
    df = pd.read_csv(_config.OUT_PLACE + 'ag4b_adjust_background/%s_fraczero_dec.csv' % (treat_nm), index_col = 0)
    df['Treatment'] = treat_nm
    df['Batch'] = exp_nm_to_batch[treat_nm]
    df['Editor'] = exp_nm_to_editor[treat_nm]
    mdf = mdf.append(df, ignore_index = True)
    timer.update()

  mdf.loc[mdf['Mean activity'] == 0, 'Mean activity'] = 1e-10
  mdf['Log10 mean activity'] = np.log10(mdf['Mean activity'])
  mdf['Mutation'] = mdf['Position'].astype(str) + mdf['Ref nt'] + mdf['Obs nt']


  print('Calculating robust rank aggregation statistics on each editor...')

  all_editor_stats = pd.DataFrame()
  timer = util.Timer(total = len(editor_set))
  for editor in editor_set:
    cond_df = mdf[mdf['Editor'] == editor]
    pv = cond_df.pivot(
      index = 'Mutation', 
      columns = 'Treatment', 
      values = 'Log10 mean activity'
    )
    pv.to_csv(out_dir + '%s_pivot_log10meanactivity.csv' % (editor))

    normalized_rank_matrix = get_normalized_rank_matrix(pv)
    # if editor == 'BE4-CP1028':
      # import code; code.interact(local=dict(globals(), **locals()))

    dd = defaultdict(list)
    for mutation in normalized_rank_matrix.index:
      pval_ub = robust_rank_aggregation_pval(mutation, normalized_rank_matrix)
      pos = int(mutation[:-2])
      ref_nt = mutation[-2]
      obs_nt = mutation[-1]
      dd['Position'].append(pos)
      dd['Ref nt'].append(ref_nt)
      dd['Obs nt'].append(obs_nt)
      dd['pval_ub'].append(pval_ub)
    stats_df = pd.DataFrame(dd)

    stats_df = stats_df.sort_values(by = 'pval_ub').reset_index()
    stats_df['Sorted index'] = stats_df.index

    fdr_threshold = 0.10
    fdr_decisions = []
    for idx, row in stats_df.iterrows():
      pval_ub = row['pval_ub']
      if False not in fdr_decisions:
        if pval_ub == -np.inf:
          fdr_decisions.append(True)
        elif pval_ub <= ((idx + 1) / len(stats_df)) * fdr_threshold:
          fdr_decisions.append(True)
        else:
          fdr_decisions.append(False)
      else:
        fdr_decisions.append(False)
    stats_df['FDR accept'] = fdr_decisions

    stats_df['Editor'] = editor
    all_editor_stats = all_editor_stats.append(stats_df, ignore_index = True)

    timer.update()

  all_editor_stats.to_csv(out_dir + 'all_editor_stats.csv')

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
def main():
  print(NAME)
  
  identify_editor_effects()

  return


if __name__ == '__main__':
  main()
