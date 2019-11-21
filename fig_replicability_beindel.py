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
inp_dir = _config.OUT_PLACE + 'fig_beindelratio_anova_c1bpfq/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

'''
  NOTE: Only fig_ script that takes as input another fig_ script.
'''


##
# Form groups
##
def get_statistics():
  '''
  '''

  master_df = pd.read_csv(inp_dir + '_batch_adjusted_all_ratios-ps0_1bpcorrect_celltype.csv', index_col = 0)

  for celltype in ['mES', 'HEK293T']:
    for editor in _data.main_base_editors:
      for library in ['12kChar', 'CtoT', 'AtoG', 'CtoGA']:

        reps = _data.get_replicates(celltype, library, editor)
        if len(reps) != 2: continue

        cond = f'{celltype}_{library}_{editor}'
        print(cond)

        df1 = master_df[master_df['Name_y'] == reps[0]]
        df2 = master_df[master_df['Name_y'] == reps[1]]

        mdf = df1.merge(df2,
          on = 'Name (unique)',
          suffixes = ['_1', '_2'],
        )

        mdf.to_csv(out_dir + f'{cond}.csv')

  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  get_statistics()

  return


if __name__ == '__main__':
  main()
