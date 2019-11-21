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
inp_dir = _config.OUT_PLACE + 'ag5a4_profile_subset/'
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
    if lib_nm == 'LibA':
      num_targets = 2000
      num_targets_per_split = 200
    elif lib_nm == 'CtoGA':
      num_targets = 4000
      num_targets_per_split = 500
    else:
      num_targets = 12000
      num_targets_per_split = 2000

    print(treat_nm)
    mdf = pd.DataFrame()
    data = None
    for start_idx in range(0, num_targets, num_targets_per_split):
      data_fn = inp_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, start_idx + num_targets_per_split - 1)
      with open(data_fn, 'rb') as f:
        temp_d = pickle.load(f)
      if data is None:
        data = temp_d
      else:
        for key in temp_d:
          data[key] = temp_d[key]

    # Data
    with open(inp_dir + '%s.pkl' % (treat_nm), 'wb') as f:
      pickle.dump(data, f)

  print('Done')
  return


if __name__ == '__main__':
  main()