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
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + '_car_analyze_cond/'

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
def get_stats():

  editors = _data.main_base_editors
  celltypes = [
    'mES',
    'HEK293T',
  ]

  for editor in editors:
    for celltype in celltypes:

      data_nm = f'{celltype}_{editor}'
      try:
        obs_df = pd.read_csv(inp_dir + f'{data_nm}_obs.csv', index_col = 0)
        pred_df = pd.read_csv(inp_dir + f'{data_nm}_model.csv', index_col = 0)
      except:
        continue

      if len(obs_df) == 0:
        continue

      JOINT_EDIT_READCOUNT_THRESHOLD = 100
      obs_df = obs_df[obs_df['joint edit count'] >= JOINT_EDIT_READCOUNT_THRESHOLD]

      obs_df['id'] = obs_df['Name'] + '_' + obs_df['pos_i'].astype(str) + ',' + obs_df['pos_j'].astype(str)
      pred_df['id'] = pred_df['Name'] + '_' + pred_df['pos_i'].astype(str) + ',' + pred_df['pos_j'].astype(str)

      mdf = obs_df.merge(pred_df, on = ['id', 'Name', 'pos_i', 'pos_j', 'Phase'], suffixes = ['_obs', '_pred'])

      mdf.to_csv(out_dir + f'{data_nm}.csv')
      print(data_nm)

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