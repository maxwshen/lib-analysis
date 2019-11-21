# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle, gzip
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + '_car_test_performance/'

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

  stats_dd = defaultdict(list)
  mdf = pd.DataFrame()
  for celltype in celltypes:
    for editor in editors:
      print(celltype, editor)
      df = None
      for tt in ['train', 'test']:
        fn = inp_dir + f'{tt}_{celltype}_{editor}.csv'
        if not os.path.isfile(fn): continue

        df = pd.read_csv(fn, index_col = 0)
        df['Celltype'] = celltype
        df['Editor'] = editor
        df['Traintest'] = tt
        df['Condition'] = f'{celltype}_{editor}'

        mdf = mdf.append(df, ignore_index = True, sort = False)

      data_fn = _config.COMBIN_MODEL_DIR + f'out/combin_data_Y_imputewt/{celltype}_12kChar_{editor}.pkl.gz'
      if not os.path.isfile(data_fn): continue
      with gzip.open(data_fn, 'rb') as f: d = pickle.load(f)
      data_len = len(d)
      num_test = len(df) if df is not None else np.nan
      stats_dd['Celltype'].append(celltype)
      stats_dd['Editor'].append(editor)
      stats_dd['Num. target sites - total'].append(data_len)
      stats_dd['Num. target sites - test'].append(num_test)
      stats_dd['Num. target sites - train + validation'].append(data_len - num_test)
      stats_dd['Num. target sites - train'].append(np.floor(0.9 * (data_len - num_test)))
      stats_dd['Num. target sites - validation'].append(np.floor(0.1 * (data_len - num_test)))

  mdf.to_csv(out_dir + f'12kChar_traintest.csv')

  pd.DataFrame(stats_dd).to_csv(out_dir + f'stats_num_targets.csv')

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