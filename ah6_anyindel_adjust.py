# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
from scipy.stats import binom

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

##
# Main iterator
##
def adjust_treatment_control(treat_nm, control_nm):
  treat_data = _data.load_data(treat_nm, 'h6_anyindel')
  control_data = _data.load_data(control_nm, 'h6_anyindel')

  treat_data = treat_data[treat_data['Total count'] >= 100]
  control_data = control_data[control_data['Total count'] >= 100]

  mdf = treat_data.merge(
    control_data, 
    on = ['Name'],
    how = 'inner',
    suffixes = ['_t', '_c'],
  )

  dd = defaultdict(list)
  for idx, row in mdf.iterrows():
    adj_val = row['Indel count_t'] - (row['Indel count_c'] / row['Total count_c']) * row['Total count_t']
    adj_val = max(0, adj_val)
    dd['Indel count adj'].append(adj_val)

  for col in dd:
    mdf[col] = dd[col]

  mdf['Indel fraction adj'] = mdf['Indel count adj'] / mdf['Total count_t']

  mdf.to_csv(out_dir + '%s.csv' % (treat_nm))

  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Function calls
  treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  timer = util.Timer(total = len(treat_control_df))
  for idx, row in treat_control_df.iterrows():
    treat_nm, control_nm = row['Treatment'], row['Control']
    adjust_treatment_control(treat_nm, control_nm)
    timer.update()

  return


if __name__ == '__main__':
  main()
