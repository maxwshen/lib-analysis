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
inp_dir = _config.OUT_PLACE + 'fulledits_ratio2_correct1bpfq_combine/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)


##
# Form groups
##
def get_statistics():
  '''
  '''

  print('Loading data of BE:indel ratios at all target sites in all conditions...')
  rdf = pd.read_csv(inp_dir + '_all_ratios.csv', index_col = 0)
  exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
  exp_design['Condition'] = exp_design['Name']
  mdf = rdf.merge(exp_design, on = 'Condition')

  mdf = mdf[mdf['Editor (no point mutation, no max)'] != 'A3A_defective']

  # Setup 
  from sklearn.preprocessing import OneHotEncoder
  oher = OneHotEncoder(sparse = False)

  editor_mat = oher.fit_transform(np.array(mdf['Editor (no point mutation, no max)']).reshape(-1, 1))
  editor_ohe_nms = oher.get_feature_names()

  '''
    Batch covariates are cell-type specific
  '''
  batch_mat = oher.fit_transform(np.array(mdf['Batch covariate']).reshape(-1, 1))
  batch_ohe_nms = oher.get_feature_names()

  feature_names = list(batch_ohe_nms) + list(editor_ohe_nms)
  ft_types = ['Batch']*len(batch_ohe_nms) + ['Editor']*len(editor_ohe_nms)
  design_mat = np.concatenate((batch_mat.T, editor_mat.T))

  '''
    Run two-way ANOVA.
    Weights are used to adjust for variable N by condition. 
  '''
  print('Fitting two-way ANOVA for batch correction of BE:indel ratios...')
  from sklearn.linear_model import LinearRegression, Ridge

  # Nx(B+E)
  X = design_mat.T

  # (Nx1)
  Y = np.array(np.log(mdf['Base edit to indel ratio']))
  W = np.array(mdf['Regression weight'])

  # model = LinearRegression(fit_intercept = False)
  model = Ridge(fit_intercept = False, alpha = 1e-10)
  model.fit(X, Y, sample_weight = W)

  # Save learned parameters
  from collections import defaultdict
  dd = defaultdict(list)
  for idx in range(len(feature_names)):
    dd['Feature name'].append(feature_names[idx])
    dd['Coefficient'].append(model.coef_[idx])
    dd['Feature type'].append(ft_types[idx])
    if ft_types[idx] == 'Batch':
      celltype = feature_names[idx].split('_')[-1]
    else:
      celltype = np.nan
    dd['Celltype'].append(celltype)

  coef_df = pd.DataFrame(dd)
  coef_df.to_csv(out_dir + 'linreg_coefs.csv')

  '''
    Batch correction using mean batch coef.
    Output: mdf -> _batch_adjusted_all_ratios-ps0_1bpcorrect_celltype.csv

    250 MB, one row per condition per target site with BE:indel ratio, adjusted BE:indel ratio, etc.
  '''
  print('Batch-adjusting BE:indel ratios...')
  batch_coefs = {batch_nm.replace('x0_', ''): val for batch_nm, val in zip(batch_ohe_nms, model.coef_[:len(batch_ohe_nms)])}
  batch_coefs_vals = list(batch_coefs.values())
  assert len(set(batch_coefs)) > 3, 'Bad learned coefficients, increase L2 penalty'

  mean_batch_val = np.mean(list(batch_coefs.values()))
  num_batches = len(batch_coefs)
  print('Num batches:', num_batches)

  adj_batch_vals = {batch_nm: mean_batch_val - batch_coefs[batch_nm] for batch_nm in batch_coefs}
  print(adj_batch_vals)

  def get_adj_be_indel_ratio(row):
    val = row['Base edit to indel ratio']
    batch = row['Batch covariate']
    return np.exp(np.log(val) + adj_batch_vals[batch])

  mdf['Batch-adjusted base edit to indel ratio'] = mdf.apply(get_adj_be_indel_ratio, axis = 'columns')
  mdf['Log10 batch-adjusted base edit to indel ratio'] = np.log10(mdf['Batch-adjusted base edit to indel ratio'])
  mdf.to_csv(out_dir + '_batch_adjusted_all_ratios-ps0_1bpcorrect_celltype.csv')

  '''
    Summarize statistics
    Output: stats_df -> _batch_adjusted_summary_stats-ps0_1bpcorrect_celltype.csv
    
    One row per editor with statistics on adjusted BE:indel ratio.
  '''
  print('Calculating summary statistics by editor...')
  from collections import defaultdict
  stats_dd = defaultdict(list)
  for editor in sorted(list(set(mdf['Editor']))):
#     print(editor)
    vals = mdf[mdf['Editor'] == editor]['Batch-adjusted base edit to indel ratio']
#     print(vals.describe())
    vals = list(vals.dropna())
    
    gmean = np.exp(np.mean(np.log(vals)))
    gstd = np.exp(np.std(np.log(vals)))

    stats_dd['Editor'].append(editor)
    stats_dd['Geometric mean'].append(gmean)
    stats_dd['Geometric std'].append(gstd)
    stats_dd['Percentile: 25th'].append(np.percentile(vals, 25))
    stats_dd['Percentile: 50th'].append(np.percentile(vals, 50))
    stats_dd['Percentile: 75th'].append(np.percentile(vals, 75))
    stats_dd['Num target sites across conditions'].append(len(vals))

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + '_batch_adjusted_summary_stats-ps0_1bpcorrect_celltype.csv')

  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  get_statistics()

  return


if __name__ == '__main__':
  main()
