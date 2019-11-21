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
inp_dir = _config.ANYEDIT_MODEL_DIR + 'out/' + 'anyedit_gbtr/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  celltypes = ['mES', 'HEK293T']

  editors = _data.main_base_editors

  mdf = pd.DataFrame()

  for editor in editors:
    for celltype in celltypes:
      fn_nm = f'{celltype}_12kChar_{editor}_regress_nonzero_model_stats'
      if not os.path.isfile(inp_dir + f'{fn_nm}.csv'): continue
      df = pd.read_csv(inp_dir + f'{fn_nm}.csv', index_col = 0)
      df['Editor'] = editor
      df['Celltype'] = celltype
      df.to_csv(out_dir + fn_nm + '.csv')
      mdf = mdf.append(df, ignore_index = True, sort = False)

    fn_nm = f'12kChar_{editor}_regress_nonzero_model_stats'
    if not os.path.isfile(inp_dir + f'{fn_nm}.csv'): continue
    df = pd.read_csv(inp_dir + f'{fn_nm}.csv', index_col = 0)
    df['Editor'] = editor
    df['Celltype'] = 'Combined'
    df.to_csv(out_dir + fn_nm + '.csv')
    mdf = mdf.append(df, ignore_index = True, sort = False)

  mdf.to_csv(out_dir + 'mdf.csv')

  return


if __name__ == '__main__':
  main()