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

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
batches = exp_design['Batch covariate']
editors = exp_design['Editor']
exp_nm_to_batch = {exp_nm: batch for exp_nm, batch in zip(exp_nms, batches)}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

##
# Model
##
def get_poswise_df(data, nm_to_seq, treat_nm):
  dd = defaultdict(list)
  timer = util.Timer(total = len(data))
  for nm in data:
    pw = data[nm]
    seq = nm_to_seq[nm]

    for jdx in range(len(pw)):
      pos = _data.idx_to_pos(jdx, treat_nm)
      ref_nt = seq[jdx]
      ref_idx = nt_to_idx[ref_nt]
      total = sum(pw[jdx])

      for kdx in range(len(pw[jdx])):
        if kdx == ref_idx:
          continue

        count = pw[jdx][kdx]

        dd['Count'].append(count)
        dd['Total count'].append(total)
        dd['Obs nt'].append(nts[kdx])
        dd['Ref nt'].append(ref_nt)
        dd['Position'].append(pos)
        dd['Name'].append(nm)

    timer.update()

  df = pd.DataFrame(dd)
  return df

def get_means(pw_df):
  pw_df = pw_df[pw_df['Total count'] >= 100]
  pw_df['Frequency'] = pw_df['Count'] / pw_df['Total count']

  pos_range = sorted(set(pw_df['Position']))

  dd = defaultdict(list)
  for pos in pos_range:
    pw_df_s1 = pw_df[pw_df['Position'] == pos]

    for ref_nt in nts:
      pw_df_s2 = pw_df_s1[pw_df_s1['Ref nt'] == ref_nt]
      
      for obs_nt in nts:
        if obs_nt == ref_nt:
          continue
        pw_df_s3 = pw_df_s2[pw_df_s2['Obs nt'] == obs_nt]
        dd['Position'].append(pos)
        dd['Ref nt'].append(ref_nt)
        dd['Obs nt'].append(obs_nt)
        dd['Mean activity'].append(np.mean(pw_df_s3['Frequency']))
  means = pd.DataFrame(dd)
  return means


def load_Y():
  '''
    Combine data together in human friendly formats 
  '''
  all_means = dict()

  print('WARNING: script depends on c_poswise_basic')
  c_fold = _config.OUT_PLACE + 'c_poswise_basic/'
  c_mtimes = [os.path.getmtime(c_fold + fn) for fn in os.listdir(c_fold)]
  ag4_fold = _config.OUT_PLACE + 'ag4_poswise_be_adjust/'
  ag4_mtimes = [os.path.getmtime(ag4_fold + fn) for fn in os.listdir(ag4_fold)]

  if min(c_mtimes) < max(ag4_mtimes):
    prompt = 'The most recent modification to a file in ag4 occurred after the earliest modification in c -- c might not be up to date. Continue? (y) '
    ans = input(prompt)
    if ans != 'y':
      sys.exit(0)

  timer = util.Timer(total = len(treat_control_df))
  for idx, row in treat_control_df.iterrows():
    timer.update()
    treat_nm = row['Treatment']
    if 'Cas9' in treat_nm:
      continue
    adj_d = _data.load_data(treat_nm, 'ag4_poswise_be_adjust')
    lib_design, seq_col = _data.get_lib_design(treat_nm)
    nms = lib_design['Name (unique)']
    seqs = lib_design[seq_col]
    nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

    # pw_df = get_poswise_df(adj_d, nm_to_seq, treat_nm)
    pw_df = pd.read_csv(_config.OUT_PLACE + 'c_poswise_basic/%s.csv' % (treat_nm))
    means = get_means(pw_df)
    means.to_csv(out_dir + 'means_%s.csv' % (treat_nm))

    all_means[treat_nm] = means

  import pickle
  with open(out_dir + 'Y.pkl', 'wb') as f:
    pickle.dump(all_means, f)

  return


def load_X():
  '''
    Combine data together in human friendly formats 
  '''

  dd = defaultdict(list)

  for idx, row in treat_control_df.iterrows():
    treat_nm = row['Treatment']
    if 'Cas9' in treat_nm:
      continue

    batch = exp_nm_to_batch[treat_nm]
    editor = exp_nm_to_editor[treat_nm]
    dd['Batch'].append(batch)
    dd['Editor'].append(editor)
    dd['Name'].append(treat_nm)

  df = pd.DataFrame(dd)
  df.to_csv(out_dir + 'X.csv')

  return

##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Function calls
  load_X()
  load_Y()

  return


if __name__ == '__main__':
  main()