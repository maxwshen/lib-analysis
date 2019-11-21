# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/home/unix/maxwshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import _data

# Default params
inp_dir = _config.OUT_PLACE + 'be_combin_12kChar_simulated_bystander/'
NAME = util.get_fn(__file__)

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)


@util.time_dec
def main():
  print(NAME)

  for idx, row in treat_control_df.iterrows():
    treat_nm = row['Treatment']
    if 'Cas9' in treat_nm:
      continue

    lib_nm = _data.get_lib_nm(treat_nm)
    if lib_nm != '12kChar': continue
    if 'U2OS' in treat_nm: continue

    num_targets = 12000
    num_targets_per_split = 2000

    print(treat_nm)
    mdf = pd.DataFrame()
    data = None
    stats_df = pd.DataFrame()
    for start_idx in range(0, num_targets, num_targets_per_split):
      stats_fn = inp_dir + '%s_%s_%s_stats.csv' % (treat_nm, start_idx, start_idx + num_targets_per_split - 1)
      df = pd.read_csv(stats_fn, index_col = 0)
      stats_df = stats_df.append(df, ignore_index = True)

    stats_df.to_csv(inp_dir + '%s.csv' % (treat_nm))

  print('Done')
  return


if __name__ == '__main__':
  main()