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
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + 'g_syn_ctoga/'

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

  celltypes = [
    'mES',
    # 'HEK293T',
  ]

  lib_to_editors = {
    'CtoGA': [
      'eA3A',
      'eA3A_T31A',
      'eA3A_T31AT44A',
      'eA3A_noUGI',
      'eA3A_noUGI_T31A',
      'eA3A_Nprime_UGId12_T30A',
    ]
  }

  for celltype in celltypes:
    for lib in lib_to_editors:
      for editor in lib_to_editors[lib]:
        inp_nm = f'{lib}_{celltype}_{editor}'

        df = pd.read_csv(inp_dir + f'{inp_nm}.csv', index_col = 0)
        df.to_csv(out_dir + f'{inp_nm}.csv')

  return


if __name__ == '__main__':
  main()