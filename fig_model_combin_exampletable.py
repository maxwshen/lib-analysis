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
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + 'f_all_cbes/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)

##
#
##
def get_stats(fn_nm):
  df = pd.read_csv(inp_dir + fn_nm, index_col = 0)

  pred_fq_cols = [col for col in df.columns if 'Predicted' in col]
  df = df.rename(columns = {col: col.split(', ')[-1] for col in pred_fq_cols})

  nt_cols = [f'C{pos}' for pos in range(-3, 15)]
  nt_cols = [col for col in nt_cols if col in df.columns]

  df = df.groupby(nt_cols).agg('sum').reset_index()
  df.to_csv(out_dir + fn_nm)

  return df

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  seq = 'GCCATCAACTCTCTATAGGATTCTAGGCTAGCGCCGTGACAGGAGGTACT'
  fn_nm = seq + '.csv'

  get_stats(fn_nm)

  return


if __name__ == '__main__':
  main()