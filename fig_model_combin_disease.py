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
inp_dir = _config.COMBIN_MODEL_DIR + 'out/' + '_car_pred_other_dis_annotate_combine/'

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
def get_stats(df, lib, editor):
  core_nt = 'A' if 'ABE' in editor else 'C'

  # # 50%
  # editor_to_core_range = {
  #   'ABE': list(range(5, 7 + 1)),
  #   'ABE-CP1040': list(range(4, 9 + 1)),
  #   'BE4': list(range(4, 8 + 1)),
  #   'BE4-CP1028': list(range(4, 13 + 1)),
  #   'AID': list(range(1, 10 + 1)),
  #   'CDA': list(range(1, 8 + 1)),
  #   'evoAPOBEC': list(range(3, 8 + 1)),
  #   'eA3A': list(range(5, 8 + 1)),
  #   'eA3A_noUGI': list(range(4, 8 + 1)),
  #   'eA3A_noUGI_T31A': list(range(4, 8 + 1)),
  #   'eA3A_Nprime_UGId12_T30A': list(range(4, 8 + 1)),
  #   'eA3Amax_T44DS45A': list(range(4, 8 + 1)),
  #   'BE4max_H47ES48A': list(range(4, 8 + 1)),
  #   'eA3A_T31A': list(range(4, 8 + 1)),
  #   'eA3A_T31AT44A': list(range(4, 8 + 1)),
  # }

  # 30%
  editor_to_core_range = {
    'ABE': list(range(4, 8 + 1)),
    'ABE-CP1040': list(range(3, 11 + 1)),
    'BE4': list(range(3, 9 + 1)),
    'BE4-CP1028': list(range(2, 15 + 1)),
    'AID': list(range(1, 11 + 1)),
    'CDA': list(range(1, 9 + 1)),
    'evoAPOBEC': list(range(2, 9 + 1)),
    'eA3A': list(range(4, 9 + 1)),
    'eA3A_noUGI': list(range(4, 8 + 1)),
    'eA3A_noUGI_T31A': list(range(4, 8 + 1)),
    'eA3A_Nprime_UGId12_T30A': list(range(4, 8 + 1)),
    'eA3Amax_T44DS45A': list(range(4, 8 + 1)),
    'BE4max_H47ES48A': list(range(4, 8 + 1)),
    'eA3A_T31A': list(range(4, 8 + 1)),
    'eA3A_T31AT44A': list(range(4, 8 + 1)),
  }

  core_range = editor_to_core_range[editor]
  min_idx = min(core_range) - 1
  max_idx = max(core_range)

  dd = defaultdict(list)
  timer = util.Timer(total = len(df))
  for idx, row in df.iterrows():
    grna = row['gRNA (20nt)']
    ct = grna[min_idx : max_idx].count(core_nt)
    dd['Num. substrate nts in editor-specific core'].append(ct)
    timer.update()

  return dd

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  for lib in ['AtoG', 'CtoT', 'CtoGA']:
    for editor in _data.main_base_editors:
      for celltype in ['mES', 'HEK293T']:

        nm = f'{lib}_{celltype}_{editor}'
        inp_fn = inp_dir + nm + '.csv'
        if not os.path.isfile(inp_fn): continue

        print(nm)

        out_fn = out_dir + f'{nm}.csv'
        # if os.path.isfile(out_fn):
        #   print('Already done, skipping')
        #   continue

        df = pd.read_csv(inp_fn, index_col = 0)

        # Annotate with editor-specific core region
        stats = get_stats(df, lib, editor)
        for col in stats:
          df[col] = stats[col]

        if 'Misannotated sequence' in df:
          df = df[~df['Misannotated sequence']]

        df.to_csv(out_fn)

  return


if __name__ == '__main__':
  main()