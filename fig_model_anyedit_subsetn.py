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
inp_dir_subsetn = _config.ANYEDIT_MODEL_DIR + 'out/' + 'anyedit_gbtr_subsetn/'
inp_dir_best = _config.ANYEDIT_MODEL_DIR + 'out/' + 'anyedit_gbtr/'

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
  celltypes = ['HEK293T', 'mES']
  libraries = ['12kChar']
  # libraries = ['CtoT', 'AtoG', 'CtoGA']

  mdf = pd.DataFrame()
  for celltype in celltypes:
    for lib_nm in libraries:
      for editor_nm in editors:

        # Get subset n
        for x_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
          for iter in range(3):

            data_nm = f'{celltype}_{lib_nm}_{editor_nm}'
            inp_fn = inp_dir_subsetn + f'{data_nm}_regress_nonzero_{x_frac}_{iter}_model_stats.csv'
            if not os.path.isfile(inp_fn): continue
            print(inp_fn)
            df = pd.read_csv(inp_fn, index_col = 0)

            df['Training fraction'] = x_frac
            df['Replicate'] = iter
            df['Editor'] = editor_nm
            df['Celltype'] = celltype
            df['Library'] = lib_nm

            mdf = mdf.append(df, ignore_index = True, sort = False)

        # Get hyperparam optimized

        data_nm = f'{celltype}_{lib_nm}_{editor_nm}'
        inp_fn = inp_dir_best + f'{data_nm}_regress_nonzero_model_stats.csv'
        if not os.path.isfile(inp_fn): continue
        print(inp_fn)
        df = pd.read_csv(inp_fn, index_col = 0)

        df['Training fraction'] = 1.1
        df['Replicate'] = 0
        df['Editor'] = editor_nm
        df['Celltype'] = celltype
        df['Library'] = lib_nm

        mdf = mdf.append(df, ignore_index = True, sort = False)


  mdf.to_csv(out_dir + f'subsetn_stats.csv')
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