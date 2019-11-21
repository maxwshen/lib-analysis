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
inp_dir = _config.OUT_PLACE + 'endo_pathogenic/'

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
# Main
##
@util.time_dec
def main():
  print(NAME)

  df = pd.read_csv(inp_dir + 'endo_pathogenic_all_stats.csv', index_col = 0)
  df.to_csv(out_dir + 'endo_pathogenic_all_stats.csv')

  return


if __name__ == '__main__':
  main()