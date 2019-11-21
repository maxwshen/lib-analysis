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
inp_dir = _config.ANYEDIT_MODEL_DIR + 'out/' + '_anyedit_pred_other_libs/'

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
def add_annotations(df):
  fe_cols = [col for col in df.columns if 'Fraction edited' in col and 'logit' not in col]
  mean_edit_fqs = df[fe_cols].apply(np.nanmean, axis = 'columns')
  df['Obs edit frequency'] = mean_edit_fqs

  from scipy.special import logit, expit
  mean_logit_edit_fq = logit(np.mean(df['Obs edit frequency']))

  # Need to choose logit std to convert data
  # std_logit_edit_fq = 1.1
  std_logit_edit_fq = 2

  df['y_pred_GBTR'] = df['Pred edit efficiency'] # col misnaming from other script
  df['Pred edit frequency'] = expit((df['y_pred_GBTR'] * std_logit_edit_fq) + mean_logit_edit_fq)


  return df

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  celltypes = ['mES', 'HEK293T']
  libs = ['AtoG', 'CtoT', 'CtoGA']
  editors = _data.main_base_editors

  mdf = pd.DataFrame()

  for editor in editors:
    for celltype in celltypes:
      for lib in libs:
        fn_nm = f'{celltype}_{lib}_{editor}'
        if not os.path.isfile(inp_dir + f'{fn_nm}.csv'): continue

        print(fn_nm)
        df = pd.read_csv(inp_dir + f'{fn_nm}.csv', index_col = 0)
        df['Editor'] = editor
        df['Celltype'] = celltype
        df = add_annotations(df)

        df.to_csv(out_dir + fn_nm + '.csv')
        mdf = mdf.append(df, ignore_index = True, sort = False)


  return


if __name__ == '__main__':
  main()