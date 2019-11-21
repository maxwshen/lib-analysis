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
inp_dir = _config.ANYEDIT_MODEL_DIR + 'out/combin_consistency_CtoGA_shortlong/'
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
  
  dd = defaultdict(list)

  editors = _data.main_base_editors
  celltypes = ['mES', 'HEK293T', 'U2OS']
  libs = ['12kChar', 'AtoG', 'CtoT', 'CtoGA']

  for treat in treat_control_df['Treatment']:
    if 'CtoGA' not in treat: continue

    inp_fn = inp_dir + f'{treat}.csv'
    # print(inp_fn)

    if os.path.isfile(inp_fn):
      command = f'cp {inp_fn} {out_dir}'
      ans = subprocess.check_output(command, shell=True)
      print(inp_fn)

  return


if __name__ == '__main__':
  main()