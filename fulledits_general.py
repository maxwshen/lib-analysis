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
inp_dir = _config.OUT_PLACE
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)


##
# Form groups
##
def form_data(exp_nm):
  '''
    Complete global editing statistics for each target site:
      - Base editing frequency among all non-noisy reads
      - Indel frequency among all non-noisy reads
    Can be used to calculate base editing to indel ratio (low biological replicability, due to low replicability of indel frequency).
  '''

  be_df = pd.read_csv(_config.OUT_PLACE + 'be_combin_general_annot/%s.csv' % (exp_nm), index_col = 0)
  indel_df = pd.read_csv(_config.OUT_PLACE + 'indel_anyindel/%s.csv' % (exp_nm), index_col = 0)

  be_df['Name'] = be_df['Name (unique)']
  indel_df = indel_df[indel_df['Total count'] >= 100]


  mdf = be_df.merge(indel_df, on = 'Name', suffixes = ('_be', '_indel'))
  mdf['Base edit freq'] = mdf['Wildtype freq'] * mdf['Fraction edited']
  mdf['Base edit to indel ratio'] = mdf['Base edit freq'] / mdf['Indel freq']

  mdf.to_csv(out_dir + '%s.csv' % (exp_nm))
  return

##
# Main
##
@util.time_dec
def main(exp_nm = ''):
  print(NAME)
  
  # Function calls
  be_treatments = [nm for nm in treat_control_df['Treatment'] if 'Cas9' not in nm]
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    form_data(treat_nm)
    timer.update()

  return


if __name__ == '__main__':
  main()