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
inp_dir = _config.OUT_PLACE + 'ag4b_adjust/'
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

def remove_ontarget_component(comb_batch):
  '''
    pool_frac_threshold = 0.25 reflects existence of sub batches
  '''
  seeds = ['A,6,G', 'C,6,T', 'C,6,G']
  pool_frac_threshold = 0.25

  for batch in comb_batch:
    df = comb_batch[batch]
    crit = (df['Pool fraction'] >= pool_frac_threshold)
    df = df[crit]

    for seed in seeds:
      visited = set()
      to_remove = [seed]
      while len(to_remove) > 0:
        mut, to_remove = to_remove[0], to_remove[1:]
        [ref_nt, pos, obs_nt] = mut.split(',')
        pos = int(pos)
        crit = (df['Position'] == pos) & (df['Obs nt'] == obs_nt) & (df['Ref nt'] == ref_nt)
        df_subset = df[crit]
        if len(df_subset) > 0:
          df = df[~crit]
          neighbor1 = '%s,%s,%s' % (ref_nt, pos - 1, obs_nt)
          neighbor2 = '%s,%s,%s' % (ref_nt, pos + 1, obs_nt)
          for neighbor in [neighbor1, neighbor2]:
            if neighbor not in to_remove and neighbor not in visited:
              to_remove.append(neighbor)
        visited.add(mut)

    comb_batch[batch] = df
    print(batch)
    print(df.sort_values(by = 'Position'))

  return comb_batch

##
# Main iterator
##
def identify_highfreq_batcheffects():

  # Gather statistics
  be_treatments = [s for s in treat_control_df['Treatment'] if 'Cas9' not in s]
  adf = dict()
  for nm in be_treatments:
    df = pd.read_csv(inp_dir + '%s_fraczero_dec.csv' % (nm), index_col = 0)
    crit = (df['Mean activity'] >= 0.01) & (df['FDR accept'] == True)
    df = df[crit]
    df = df[['Position', 'Obs nt', 'Ref nt']]
    adf[nm] = df

  batch_to_nms = defaultdict(list)
  editor_to_nms = defaultdict(list)
  for nm in be_treatments:
    batch = exp_nm_to_batch[nm]
    batch_to_nms[batch].append(nm)

    editor = exp_nm_to_editor[nm]
    editor_to_nms[editor].append(nm)

  batches = set(batch_to_nms.keys())
  editors = set(editor_to_nms.keys())

  comb_batch = get_combined_grid_profiles(adf, batches, batch_to_nms)
  comb_editor = get_combined_grid_profiles(adf, editors, editor_to_nms)

  import pickle
  with open(out_dir + 'comb_batch.pkl', 'wb') as f:
    pickle.dump(comb_batch, f)

  with open(out_dir + 'comb_editor.pkl', 'wb') as f:
    pickle.dump(comb_editor, f)


  '''
    Identify high frequency batch effects.
    Note: This procedure is highly sensitive to parameters

    In batch profiles, remove connected component at A6 and C6
    Identify remaining mutations with batch frequency >50%

    Highly recommended to use 
    ag4e_highfreq_batcheffects-show_filter.ipynb
    to visualize which mutations are being filtered

  '''
  comb_batch = remove_ontarget_component(comb_batch)
  
  with open(out_dir + 'comb_batch_filtontarget.pkl', 'wb') as f:
    pickle.dump(comb_batch, f)


  # Load pws and remove
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''
  print('Removing high frequency batch effects in each condition...')
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    adj_d = pickle.load(open(inp_dir + '%s.pkl' % (treat_nm), 'rb'))
    lib_design, seq_col = _data.get_lib_design(treat_nm)
    nms = lib_design['Name (unique)']
    seqs = lib_design[seq_col]
    nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

    batch = exp_nm_to_batch[treat_nm]
    to_remove = comb_batch[batch]
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
  identify_highfreq_batcheffects()


  return


if __name__ == '__main__':
  main()

