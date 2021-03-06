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
inp_dir = _config.OUT_PLACE + 'fulledits_ratio2_predict/'

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
  if 'Num' in col:
    return col
  idx = int(col.split('_')[-1][:-1])
  ft_nm = col.split('_')[-1][-1]
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

  fn = f'{cond}_model_stats.csv'
  df = pd.read_csv(inp_dir + fn, index_col = 0)

  po_fn = f'{cond}_df.csv'
  po_df = pd.read_csv(inp_dir + po_fn, index_col = 0)
  df['Num. substrate nucleotides'] = len(po_df)
  df['Num. substrate nucleotides - train'] = sum(po_df['TrainTest_LinearReg'] == 'train')
  df['Num. substrate nucleotides - test'] = sum(po_df['TrainTest_LinearReg'] == 'test')

  from scipy.stats import pearsonr
  po_dfs = po_df[po_df['TrainTest_LinearReg'] == 'test']
  d1 = po_df['y_pred_LinearReg']
  d2 = po_df['y']
  pr, pval = pearsonr(d1, d2)
  df['LinearReg pearsonr test pval'] = pval

  df['Condition'] = cond

  ft_cols = [col for col in df.columns if col[0] == 'x']
  pos1_idx = 1
  ft_col_renamer = {ft_col: feature_idx_to_pos(ft_col, pos1_idx) for ft_col in ft_cols}
  print(ft_col_renamer)
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
    'BE4-CP1028': '190419_mES_12kChar_BE4-CP1028',
    'evoAPOBEC': '190419_mES_12kChar_evoAPOBEC',
    'AID': '190329_mES_12kChar_AID',
    'CDA': '190418_mES_12kChar_CDA',
    'BE4': '190307_mES_12kChar_BE4',
    'eA3A': '190418_mES_12kChar_eA3A',
    'ABE-CP1044': '190418_mES_12kChar_ABE-CP1044',
    'ABE': '190307_mES_12kChar_ABE',
  }

  for editor in editor_to_cond:
    cond = editor_to_cond[editor]
    print(editor, cond)
    get_stats(cond, editor)

  return


if __name__ == '__main__':
  main()