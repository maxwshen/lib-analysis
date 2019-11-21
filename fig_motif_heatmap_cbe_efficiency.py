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
def get_stats(cond, new_nm, task):

  fn = f'{cond}_{task}_regress_nonzero_model_stats.csv'
  if not os.path.isfile(inp_dir + fn): return
  df = pd.read_csv(inp_dir + fn, index_col = 0)

  po_fn = f'{cond}_{task}_regress_nonzero_df.csv'
  po_df = pd.read_csv(inp_dir + po_fn, index_col = 0)
  df['Num. substrate nucleotides'] = len(po_df)
  df['Num. substrate nucleotides - train'] = sum(po_df['TrainTest_LinearReg'] == 'train')
  df['Num. substrate nucleotides - test'] = sum(po_df['TrainTest_LinearReg'] == 'test')

  df['Condition'] = cond

  ft_cols = [col for col in df.columns if col[0] == 'x' and 'x20' not in col]
  pos1_idx = 10
  ft_col_renamer = {ft_col: feature_idx_to_pos(ft_col, pos1_idx) for ft_col in ft_cols}
  df = df.rename(columns = ft_col_renamer)

  df.to_csv(out_dir + f'{new_nm}_{task}.csv')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  public_exp_design = _data.get_public_lib_design()

  task = 'C_TGA'
  timer = util.Timer(total = len(public_exp_design))
  for idx, row in public_exp_design.iterrows():
    cond = row['Name']
    new_nm = row['New name']

    get_stats(cond, new_nm, task)
    timer.update()

  return


if __name__ == '__main__':
  main()