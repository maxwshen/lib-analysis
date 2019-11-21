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

info_cols = ['Name', 'Category', 'Indel start', 'Indel end', 'Indel length', 'MH length', 'Inserted bases']


##
# Statistics
##
def gather_stats_binom_control_muts(t, c, seq, treat_nm, nm, decisions):
  '''
    Filter treatment mutations that can be explained by control freq.
    In practice, this step is most effective for control mutations
    with relatively high frequency => relatively high variance

    Considers all events that occur (fq > 0%) in both control and treatment data
  '''
  fpr_threshold_try1 = 0.10

  # Detect if no indels in control
  c_cat_set = set(c['Category'])
  if len(c_cat_set) == 1 and 'wildtype' in c_cat_set:
    return
  t_cat_set = set(t['Category'])
  if len(t_cat_set) == 1 and 'wildtype' in t_cat_set:
    return

  t['Frequency'] = t['Count'] / sum(t['Count'])
  c['Frequency'] = c['Count'] / sum(c['Count'])

  mdf = t.merge(c, on = info_cols, suffixes = ('_t', '_c'))
  t_tot = sum(mdf['Count_t'])

  for idx, row in mdf.iterrows():
    if row['Category'] == 'wildtype':
      continue
    t_count = row['Count_t']
    c_fq = row['Frequency_c']
    pval = binom.sf(t_count - 1, t_tot, c_fq)

    decisions['pval'].append(pval)
    decisions['nm'].append(nm)
    for col in info_cols:
      decisions[col].append(row[col])

  return


##
# Filters
##
def filter_high_control_muts(t, c, seq, treat_nm, nm, decisions):
  '''
    Filter positions with very high control mut freq. 
    with significant support from readcounts
  '''
  max_control_mut_fq = 0.05


  # Detect if no indels in control
  c_cat_set = set(c['Category'])
  if len(c_cat_set) == 1 and 'wildtype' in c_cat_set:
    return t

  c['Frequency'] = c['Count'] / sum(c['Count'])

  num_hf = 0
  max_hf_freq = 0
  for idx, row in c.iterrows():
    if row['Category'] == 'wildtype':
      continue
    fq = row['Frequency']
    if fq >= max_control_mut_fq:
      num_hf += 1
    if fq >= max_hf_freq:
      max_hf_freq = fq

  decisions['nm'].append(nm)
  decisions['wiped'].append(bool(num_hf > 0))
  decisions['num_hf'].append(num_hf)
  decisions['max_hf_freq'].append(max_hf_freq)

  '''
    If any high frequency control indels, disregard all treatment indel data.
  '''
  if num_hf > 0:
    return None
  else:
    return t


def filter_binom_control_muts(to_remove, adj_d, control_data, nm_to_seq):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    nm = row['nm']
    t = adj_d[nm]

    # info_cols = ['Category', 'Indel start', 'Indel end', 'Indel length', 'MH length', 'Inserted bases']
    for jdx, trow in t.iterrows():
      crit_match = bool(sum([bool(trow[col] == row[col]) for col in info_cols]) == len(info_cols))
      if crit_match:
        trow['Count'] = 0

    t = t[t['Count'] > 0]

    adj_d[nm] = t
    timer.update()

  return adj_d


##
# Main iterator
##
def adjust_treatment_control(treat_nm, control_nm):
  treat_data = _data.load_data(treat_nm, 'h6_anyindel')
  control_data = _data.load_data(control_nm, 'h6_anyindel')

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  ''' 
    h6 format: data is a dict, keys = target site names
    values = dfs
      'Category', 
      'Indel start', 
      'Indel end', 
      'Indel length', 
      'MH length',
      'Inserted bases',
      'Count',
      'Name',
  '''

  adj_d = dict()
  stats_dd = defaultdict(list)

  '''
    Filter positions with abnormally high control mut freq. 
  '''
  hc_decisions = defaultdict(list)
  print('Filtering positions with high frequency control mutations...')
  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    timer.update()
    nm = row['Name (unique)']
    seq = row[seq_col]

    stats_dd['Name'].append(nm)

    if nm not in treat_data:
      stats_dd['Status'].append('No treatment')
      continue

    t = treat_data[nm]
    if nm not in control_data:
      stats_dd['Status'].append('No control')
      adj_d[nm] = t
      continue

    stats_dd['Status'].append('Adjusted')
    c = control_data[nm]

    # Adjust
    t = filter_high_control_muts(t, c, seq, treat_nm, nm, hc_decisions)
    if t is not None:
      adj_d[nm] = t

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + '%s_stats.csv' % (treat_nm))

  hc_df = pd.DataFrame(hc_decisions)
  hc_df = hc_df.sort_values(by = 'max_hf_freq', ascending = False)
  hc_df = hc_df.reset_index(drop = True)
  hc_df.to_csv(out_dir + '%s_hc_dec.csv' % (treat_nm))

  '''
    Filter treatment mutations that can be explained by control freq.
    In practice, this step is most effective for control mutations
    with relatively high frequency => relatively high variance
  '''
  print('Gathering statistics on treatment mutations explained by control mutations...')
  bc_decisions = defaultdict(list)
  timer = util.Timer(total = len(adj_d))
  for nm in adj_d:
    t = adj_d[nm]
    if nm not in control_data:
      continue
    c = control_data[nm]
    seq = nm_to_seq[nm]

    gather_stats_binom_control_muts(t, c, seq, treat_nm, nm, bc_decisions)
    timer.update()

  '''
    Using global statistics, filter mutations
    while controlling false discovery rate
  '''
  bc_fdr_threshold = 0.05
  bc_df = pd.DataFrame(bc_decisions)
  other_distribution = bc_df[bc_df['pval'] > 0.995]
  bc_df = bc_df[bc_df['pval'] <= 0.995]
  bc_df = bc_df.sort_values(by = 'pval')
  bc_df = bc_df.reset_index(drop = True)

  fdr_decs, hit_reject = [], False
  for idx, pval in enumerate(bc_df['pval']):
    if hit_reject:
      dec = False
    else:
      fdr_critical = ((idx + 1) / len(bc_df)) * bc_fdr_threshold
      dec = bool(pval <= fdr_critical)
    fdr_decs.append(dec)
    if dec is False and hit_reject is True:
      hit_reject = False
  bc_df['FDR accept'] = fdr_decs

  other_distribution['FDR accept'] = False
  bc_df = bc_df.append(other_distribution, ignore_index = True)
  bc_df.to_csv(out_dir + '%s_bc_dec.csv' % (treat_nm))

  print('Filtering treatment mutations explained by control mutations...')
  to_remove = bc_df[bc_df['FDR accept'] == False]
  adj_d = filter_binom_control_muts(to_remove, adj_d, control_data, nm_to_seq)

  ##
  # Write
  ##
  with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
    pickle.dump(adj_d, f)

  return

##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  # Generate qsubs only for unfinished jobs
  treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  num_scripts = 0
  for idx, row in treat_control_df.iterrows():
    treat_nm, control_nm = row['Treatment'], row['Control']

    if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
      if os.path.getsize(out_dir + '%s.pkl' % (treat_nm)) > 0:
        continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    '''
      Empirically determined
      pickle > 37 mb: needs 4 gb ram
      pickle > 335 mb: needs 8 gb ram
    '''
    print(treat_nm)
    mb_file_size = _data.check_file_size(treat_nm, 'h6_anyindel')
    ram_gb = 2
    if mb_file_size > 30:
      ram_gb = 4
    if mb_file_size > 300:
      ram_gb = 8
    if mb_file_size > 1000:
      ram_gb = 16

    '''
      Can be very slow - up to 8h+ for some conditions.

      Could help to split 3 steps into 3 scripts.
      Statistical tests should be performed globally (for accurate FDR thresholds), and luckily these are the fast parts of the pipeline

      Subtracting control from treatment involves a lot of dataframe manipulations and is the bottleneck step. Fortunately, this can be parallelized
    '''

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, control_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=16:00:00,h_vmem=%sG -wd %s %s &' % (ram_gb, _config.SRC_DIR, sh_fn))

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return

##
# Main
##
@util.time_dec
def main(treat_nm = '', control_nm = ''):
  print(NAME)
  
  # Function calls
  adjust_treatment_control(treat_nm, control_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1], control_nm = sys.argv[2])
  else:
    gen_qsubs()
