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
inp_dir = _config.OUT_PLACE + 'be_rarefrac_CBE_models/'

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
def get_stats(cond, editor, task):

  fn = f'{cond}_{task}_regress_nonzero_model_stats.csv'
  df = pd.read_csv(inp_dir + fn, index_col = 0)

  po_fn = f'{cond}_{task}_regress_nonzero_df.csv'
  po_df = pd.read_csv(inp_dir + po_fn, index_col = 0)
  df['Num. substrate nucleotides'] = len(po_df)
  model_nm = 'LinearReg'
  train_df = po_df[po_df[f'TrainTest_{model_nm}'] == 'train']
  test_df = po_df[po_df[f'TrainTest_{model_nm}'] == 'test']
  df['Num. substrate nucleotides - train'] = len(train_df)
  df['Num. substrate nucleotides - test'] = len(test_df)
  df['Num. targets'] = len(set(po_df['Name']))
  df['Num. targets - train'] = len(set(train_df['Name']))
  df['Num. targets - test'] = len(set(test_df['Name']))

  df['Condition'] = cond

  ft_cols = [col for col in df.columns if col[0] == 'x' and 'x20' not in col]
  pos1_idx = 10
  ft_col_renamer = {ft_col: feature_idx_to_pos(ft_col, pos1_idx) for ft_col in ft_cols}
  df = df.rename(columns = ft_col_renamer)

  df.to_csv(out_dir + f'{editor}_{task}.csv')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  tasks = [
    'C_GA_over_C_D',
    'C_T_over_C_D',
    'C_G_over_C_D',
    'C_A_over_C_D',
  ]

  editor_to_cond = {
    'BE4-CP1028': '190419_mES_12kChar_BE4-CP1028',
    'evoAPOBEC': '190419_mES_12kChar_evoAPOBEC',
    'AID': '190329_mES_12kChar_AID',
    'CDA': '190418_mES_12kChar_CDA',
    'BE4': '190307_mES_12kChar_BE4',
    'eA3A': '190418_mES_12kChar_eA3A',
    'BE4max_H47ES48A': '190729_HEK293T_12kChar_BE4max_H47ES48A',
    'eA3Amax_T44DS45A': '190729_HEK293T_12kChar_eA3Amax_T44DS45A',
    'eA3A_noUGI': '190801_HEK293T_12kChar_eA3A_noUGI',
    'eA3A_noUGI_T31A': '190801_HEK293T_12kChar_eA3A_noUGI_T31A',
    'eA3A_T31A': '190822_mES_12kChar_eA3Amax_T31A_NG_r1',
    'eA3A_T31AT44A': '190822_mES_12kChar_eA3Amax_T31AT44A_NG_r1',
  }

  for editor in editor_to_cond:
    for task in tasks:
      cond = editor_to_cond[editor]
      print(editor, cond, task)
      get_stats(cond, editor, task)

  return


if __name__ == '__main__':
  main()