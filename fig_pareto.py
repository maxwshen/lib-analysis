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
inp_dir_simbys = _config.OUT_PLACE + 'be_combin_12kChar_simulated_bystander/'
inp_dir_mutant = _config.OUT_PLACE + 'mutant_ctog/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)

##
# Helper
##
def load_transversion_data(nm_to_conds):
  nms = list(nm_to_conds.keys())

  df = pd.read_csv(inp_dir_mutant + f'mmdf_12kChar.csv')
  combined_conds = [f'C_GA_{cond}' for cond in sorted(nms)]
  id_cols = ['Name', 'Position']
  dfs = df[id_cols + combined_conds]

  dfs.to_csv(out_dir + 'transversion_purity.csv')

  # Bootstrap
  bs_dd = dict()
  timer = util.Timer(total = len(nms))
  for nm in nms:
    col = f'C_GA_{nm}'
    data = dfs[col].dropna()

    bs_means = []
    for bs_idx in range(5000):
      bs_data = np.random.choice(data, size = len(data))
      bs_means.append(np.mean(bs_data))
    bs_dd[nm] = bs_means
    timer.update()

  bs_df = pd.DataFrame(bs_dd)
  bs_df.to_csv(out_dir + 'transversion_purity-bootstrap.csv')

  return

def load_simulated_bystander_data(nm_to_conds):
  # Combine replicates

  lib_design = pd.read_csv(_config.DATA_DIR + 'library_12kChar.csv', index_col = 0)

  ddf = dict()
  bs_dd = dict()

  timer = util.Timer(total = len(nm_to_conds))
  for nm in nm_to_conds:
    conds = nm_to_conds[nm]
    assert len(conds) == 2

    d1 = pd.read_csv(inp_dir_simbys + f'{conds[0]}.csv', index_col = 0)
    d2 = pd.read_csv(inp_dir_simbys + f'{conds[1]}.csv', index_col = 0)

    val_cols = [
      'Simulated bystander precision at editor-specific central nt',
      'Edited count',
    ]
    id_cols = [col for col in d1.columns if col not in val_cols]

    mdf = d1.merge(
      d2, 
      on = id_cols, 
      suffixes = [
        f'_1',
        f'_2',
      ]
    )

    val_col = 'Simulated bystander precision at editor-specific central nt'
    mdf[f'{val_col}'] = mdf[[f'{val_col}_1', f'{val_col}_2']].apply(np.nanmean, axis = 'columns')

    val_col = 'Edited count'
    mdf[f'{val_col}'] = mdf[[f'{val_col}_1', f'{val_col}_2']].apply(np.sum, axis = 'columns')

    mdf = mdf[mdf['Edited count'] > 100]

    ddf[nm] = mdf

    # Bootstrap
    val_col = 'Simulated bystander precision at editor-specific central nt'
    data = mdf[val_col].dropna()

    bs_means = []
    for bs_idx in range(5000):
      bs_data = np.random.choice(data, size = len(data))
      bs_means.append(np.mean(bs_data))
    bs_dd[nm] = bs_means

    timer.update()

  sim_bys_col = 'Simulated bystander precision at editor-specific central nt'
  for nm in ddf:
    df = ddf[nm]
    df = df[[
      'Name (unique)', 
      sim_bys_col, 
      'Simulated bystander position', 
      'Simulated bystander position, distance to center',
    ]]
    df = df.rename(columns = {
      sim_bys_col: f'{sim_bys_col}, {nm}',
      'Simulated bystander position': f'Simulated bystander position, {nm}',
      'Simulated bystander position, distance to center': f'Simulated bystander position, distance to center, {nm}',
    })
    lib_design = lib_design.merge(df, on = 'Name (unique)', how = 'outer')

  lib_design.to_csv(out_dir + 'simulated_bystander_data.csv')

  bs_df = pd.DataFrame(bs_dd)
  bs_df.to_csv(out_dir + 'simulated_bystander_data-bootstrap.csv')

  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  nm_to_conds = {
    'HEK293T_12kChar_eA3Amax': [
      '190103_HEK293T_12kChar_eA3A',
      '190427_HEK293T_12kChar_eA3A',
    ], 
    'HEK293T_12kChar_BE4max': [
      '190103_HEK293T_12kChar_BE4',
      '190419_HEK293T_12kChar_BE4',
    ], 
    'HEK293T_12kChar_evoAPOBECmax': [
      '190103_HEK293T_12kChar_evoAPOBEC',
      '190429_HEK293T_12kChar_evoAPOBEC',
    ], 
    'HEK293T_12kChar_AIDmax': [
      '190103_HEK293T_12kChar_AID',
      '190429_HEK293T_12kChar_AID',
    ], 
    'HEK293T_12kChar_CDAmax': [
      '190103_HEK293T_12kChar_CDA',
      '190429_HEK293T_12kChar_CDA',
    ], 
    'HEK293T_12kChar_CP1028_BE4max': [
      '190103_HEK293T_12kChar_BE4-CP1028',
      '190429_HEK293T_12kChar_BE4-CP',
    ],
    'HEK293T_12kChar_BE4max_H47ES48A': [
      '190621_HEK293T_12kChar_6kv5_BE4max_H47ES48A',
      '190729_HEK293T_12kChar_BE4max_H47ES48A',
    ],
    'HEK293T_12kChar_eA3Amax_T44DS45A': [
      '190621_HEK293T_12kChar_6kv5_eA3Amax_T44DS45A',
      '190729_HEK293T_12kChar_eA3Amax_T44DS45A',
    ],
  }

  load_simulated_bystander_data(nm_to_conds)
  load_transversion_data(nm_to_conds)
  
  return


if __name__ == '__main__':
  main()