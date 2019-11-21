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
inp_dir = _config.OUT_PLACE + 'fulledits_ratio/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  import glob
  mdf = pd.DataFrame()
  fns = glob.glob(inp_dir + '*bootstrap*')
  for fn in fns:
    cond = fn.split('/')[-1].replace('_bootstrap.csv', '')
    df = pd.read_csv(fn, index_col = 0)
    df['Condition'] = cond
    mdf = mdf.append(df, ignore_index = True)

  mdf.to_csv(out_dir + '_combined_gmean_bootstrap.csv')

  return


if __name__ == '__main__':
  main()