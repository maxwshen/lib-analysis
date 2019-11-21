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
inp_dir = _config.OUT_PLACE + 'ag4a2_adjust_batch_effects/'
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
batch_to_exp_nms = {batch: [exp_nm for exp_nm in exp_nms if exp_nm_to_batch[exp_nm] == batch] for batch in batches}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

editor_set = set(editors)
editor_set.remove('UT')
editor_set.remove('Cas9')


##
# Filters
##
def filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    pos_idx = _data.pos_to_idx(row['Position'], treat_nm)
    kdx = nt_to_idx[row['Obs nt']]
    ref_nt = row['Ref nt']
    
    for nm in adj_d:
      seq = nm_to_seq[nm]
      if pos_idx >= len(seq):
        continue
      if seq[pos_idx] == ref_nt:
        t = adj_d[nm]
        t[pos_idx][kdx] = np.nan
        adj_d[nm] = t
    timer.update()

  return adj_d


##
# Main iterator
##
def filter_inprofile_batch_effects():
  df = pd.read_csv(_config.DATA_DIR + 'batch_effects.csv')
  inprofile_batches = set(df['Batch'])

  be_treatments = [s for s in treat_control_df['Treatment'] if 'Cas9' not in s]
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    batch = exp_nm_to_batch[treat_nm]

    if batch in inprofile_batches:
      print(treat_nm, batch)
      adj_d = _data.load_data(treat_nm, 'ag4a2_adjust_batch_effects')
      to_remove = df[df['Batch'] == batch]
 
      lib_design, seq_col = _data.get_g4_lib_design(treat_nm)
      nms = lib_design['Name (unique)']
      seqs = lib_design[seq_col]
      nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

      adj_d = filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm)
      with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
        pickle.dump(adj_d, f)

    else:
      inp_fn = inp_dir + '%s.pkl' % (treat_nm)
      subprocess.check_output(
        'cp %s %s' % (inp_fn, out_dir),
        shell = True
      )

    timer.update()

  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  filter_inprofile_batch_effects()

  return


if __name__ == '__main__':
  main()
