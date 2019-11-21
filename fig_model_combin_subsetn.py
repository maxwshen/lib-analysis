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
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + '_c_log_to_df/'

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
def parse_id(df):
  hps, fracs = [], []
  for idx, row in df.iterrows():
    nm = row['Hyperparam ID']
    if 'frac' in nm:
      hp = '_'.join(nm.split('_')[:-1])
      frac = nm.split('_')[-1]
    else:
      hp = nm
      frac = 'frac1.0'
    hps.append(hp)
    fracs.append(frac)
  return hps, fracs

def form_long_df(df):
  dd = defaultdict(list)

  for idx, row in df.iterrows():
    for idx_nm in range(0, 3):
      dd['Valid loss'].append(row[str(idx_nm)])
      dd['Replicate'].append(idx_nm)
      dd['Dataset ID'].append(row['Dataset ID'])
      dd['Training fraction'].append(float(row['Training fraction'].replace('frac', '')))
      dd['Hyperparam'].append(row['Hyperparam'])
  return pd.DataFrame(dd)

##
#
##
def get_stats():

  df = pd.read_csv(inp_dir + f'_stats_validation_pivot.csv')

  is_keep = lambda row: bool('190609' in row['Hyperparam ID']) or bool(row['Hyperparam ID'] == '190524_enc_64x2_dec_256x2_nc_r5_nopos_d0.05')
  is_mes = lambda row: bool('mES' in row['Dataset ID'])
  df = df[df.apply(is_keep, axis = 'columns')]
  df = df[df.apply(is_mes, axis = 'columns')]

  hps, fracs = parse_id(df)
  df['Hyperparam'] = hps
  df['Training fraction'] = fracs

  long_df = form_long_df(df)

  long_df.to_csv(out_dir + f'subsetn_stats.csv')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  get_stats()

  return


if __name__ == '__main__':
  main()