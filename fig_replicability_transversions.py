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
endo_design = pd.read_csv(_config.DATA_DIR + 'endo_design.csv', index_col = 0)


##
# 
##
def get_replicability_stats(rep1, rep2, celltype, editor):
  for mut_nm in ['C_A_over_C_D', 'C_G_over_C_D']:
    df1 = pd.read_csv(inp_dir + f'{rep1}_{mut_nm}_regress_nonzero_df.csv', index_col = 0)
    df2 = pd.read_csv(inp_dir + f'{rep2}_{mut_nm}_regress_nonzero_df.csv', index_col = 0)

    mdf = df1.merge(df2, 
      on = [
        'Name',
        'Position',
        'MutName',
        'Local context',
      ],
      suffixes = ['_1', '_2']
    )
    print(mdf.shape)
    if len(mdf) == 0: continue
    mdf.to_csv(out_dir + f'{celltype}_{editor}_12kChar_{mut_nm}.csv')
  return

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  editors = [
    'eA3A',
    'eA3A_noUGI',
    'eA3A_Nprime_UGId12_T30A',
    'eA3A_noUGI_T31A',
  ]
  celltypes = ['mES', 'HEK293T']

  for editor in editors:
    for celltype in celltypes:
      reps = _data.get_replicates(celltype, '12kChar', editor)
      if len(reps) != 2: continue

      print(celltype, editor)

      get_replicability_stats(reps[0], reps[1], celltype, editor)


  return


if __name__ == '__main__':
  main()