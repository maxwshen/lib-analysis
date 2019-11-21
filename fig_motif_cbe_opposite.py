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
inp_dir = _config.OUT_PLACE + 'be_efficiency_CBE_models/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)


##
# Rename feature cols
##
def feature_idx_to_pos(col, pos1_idx):
  idx = int(col[1:].split('_')[0])
  ft_nm = ' '.join(col[1:].split('_')[1:])
  if idx < pos1_idx:
    pos = idx - pos1_idx
    sign = ''
  else:
    pos = idx - (pos1_idx - 1)
    sign = '+'
  return f'x{sign}{pos}{ft_nm}'


##
# Form groups
##
def get_stats(cond, editor):

  fn = f'{cond}_G_A_classify_zero_model_stats.csv'
  df = pd.read_csv(inp_dir + fn, index_col = 0)

  po_fn = f'{cond}_G_A_classify_zero_df.csv'
  po_df = pd.read_csv(inp_dir + po_fn, index_col = 0)
  df['Num. substrate nucleotides'] = len(po_df)

  model_nm = 'LogisticReg'
  train_df = po_df[po_df[f'TrainTest_{model_nm}'] == 'train']
  test_df = po_df[po_df[f'TrainTest_{model_nm}'] == 'test']
  df['Num. substrate nucleotides - train'] = len(train_df)
  df['Num. substrate nucleotides - test'] = len(test_df)
  df['Num. targets'] = len(set(po_df['Name']))
  df['Num. targets - train'] = len(set(train_df['Name']))
  df['Num. targets - test'] = len(set(test_df['Name']))

  dfs = po_df[po_df['TrainTest_LogisticReg'] == 'test']
  y0 = dfs[dfs['y'] == 0]['y_pred_LogisticReg']
  y1 = dfs[dfs['y'] == 1]['y_pred_LogisticReg']
  from scipy.stats import mannwhitneyu

  mwu_stat, pval = mannwhitneyu(y0, y1)
  df['Mann-Whitney U statistic'] = mwu_stat
  df['Mann-Whitney U p val'] = pval

  df['Condition'] = cond

  ft_cols = [col for col in df.columns if col[0] == 'x']
  pos1_idx = 6
  ft_col_renamer = {ft_col: feature_idx_to_pos(ft_col, pos1_idx) for ft_col in ft_cols}
  df = df.rename(columns = ft_col_renamer)

  df.to_csv(out_dir + f'{editor}.csv')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  editor_to_cond = {
    'BE4': '190307_HEK_CtoT_BE4',
    'BE4-CP1028': '190103_HEK293T_CtoT_BE4-CP1028',
    'evoAPOBEC': '190103_HEK293T_CtoT_evoAPOBEC',
    'AID': '190329_mES_12kChar_AID',
  }

  for editor in editor_to_cond:
    cond = editor_to_cond[editor]
    print(editor, cond)
    get_stats(cond, editor)

  return


if __name__ == '__main__':
  main()