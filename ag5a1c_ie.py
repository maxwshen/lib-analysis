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
inp_dir = _config.OUT_PLACE + 'ag5a1b_subtract/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

##
# Statistics
##
def binom_minus_binom_pval(obs_count, t_bin_p, c_bin_p, tot):
  obs_count = int(np.floor(obs_count))

  ptot = 0
  p = 1
  idx = 0
  while p > 1e-6:
    # ex: p(control = 0 and treatment >= 4) + ...
    # sf (survival function) = 1 - cdf, where cdf is p(<= threshold)
    treatment_tail = binom.sf(idx + obs_count - 1, tot, t_bin_p)
    p = binom.pmf(idx, tot, c_bin_p) * treatment_tail
    ptot += p
    idx += 1

  return ptot

def get_poswise_stats(df, mut_cols):
  stats = defaultdict(lambda: 0)
  for idx, row in df.iterrows():
    count = row['Count']
    for col in mut_cols:
      obs_nt = row[col]
      stats['%s_%s' % (col, obs_nt)] += count
      stats['%s_tot' % (col)] += count
  return stats



def gather_stats_illumina_errors(t, c, t_minq, c_minq, seq, treat_nm, nm, decisions):
  '''
    Identify mutations explainable by Illumina sequencing error
    Filtered at Q30 (1e-3), most columns have minimum
    Q = 32 (6e-4), or
    Q = 36 (2e-4)

    Considers all events (>0 freq.) in treatment data.
  '''
  fpr_threshold = 0.05
  mut_cols = [s for s in t.columns if s not in ['Count']]

  stats_t = get_poswise_stats(t, mut_cols)

  for col in mut_cols:
    ref_nt = col[0]
    t_tot = stats_t['%s_tot' % (col)]
    if t_tot == 0:
      continue

    try:
      t_bin_p = 10**(-t_minq[col] / 10)
    except KeyError:
      # Occurs at columns with all .
      t_bin_p = 10**(-30/10)

    try:
      c_bin_p = 10**(-c_minq[col] / 10)
    except KeyError:
      c_bin_p = 10**(-30/10)

    for obs_nt in list('ACGT'):
      if obs_nt == ref_nt:
        continue

      t_count = stats_t['%s_%s' % (col, obs_nt)]

      if t_count > 0:
        t_fq = t_count / t_tot

        pval = binom_minus_binom_pval(t_count, t_bin_p, c_bin_p, t_tot)

        decisions['obs_nt'].append(obs_nt)
        decisions['ref_nt'].append(ref_nt)
        decisions['t_bin_p'].append(t_bin_p)
        decisions['c_bin_p'].append(c_bin_p)
        decisions['t_ct'].append(t_count)
        decisions['t_fq'].append(t_fq)
        decisions['t_tot'].append(t_tot)
        decisions['col'].append(col)
        decisions['pos'].append(col[1:])
        decisions['pval'].append(pval)
        decisions['nm'].append(nm)
  return


##
# Filters
##
def filter_illumina_error_muts(to_remove, adj_d, control_data, nm_to_seq):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    nm = row['nm']

    t = adj_d[nm]
    c = control_data[nm]
    seq = nm_to_seq[nm]

    col_nm = row['col']
    obs_nt = row['obs_nt']

    ref_nt = col_nm[0]

    # t.loc[t[col_nm] == obs_nt, col_nm] = ref_nt

    for idx, row in t.iterrows():
      if row[col_nm] == obs_nt:
        row[col_nm] = ref_nt

    adj_d[nm] = t
    timer.update()

  return adj_d

##
# Main iterator
##
def adjust_treatment_control(treat_nm, control_nm):
  adj_d = _data.load_data(treat_nm, 'ag5a1b_subtract')
  control_data = _data.load_data(control_nm, 'g5_combin_be')

  treat_minq = _data.load_minq(treat_nm, 'g5_combin_be')
  control_minq = _data.load_minq(control_nm, 'g5_combin_be')

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  ''' 
    g5 format: data is a dict, keys = target site names
    values = dfs, with columns as '%s%s' % (nt, position), and 'Count' column
  '''

  '''
    Filter treatment mutations that are best explained by 
    spontaneous random mutations.
    Tend to be very low frequency with no counterpart in control
  '''
  print('Gathering statistics on treatment mutations explained by Illumina sequencing errors...')
  ie_decisions = defaultdict(list)
  timer = util.Timer(total = len(adj_d))
  for nm in adj_d:
    t = adj_d[nm]
    if nm not in control_data:
      continue
    c = control_data[nm]
    seq = nm_to_seq[nm]
    c_minq = control_minq[nm]
    t_minq = treat_minq[nm]

    gather_stats_illumina_errors(t, c, t_minq, c_minq, seq, treat_nm, nm, ie_decisions)
    timer.update()

  ie_fdr_threshold = 0.05
  ie_df = pd.DataFrame(ie_decisions)
  other_distribution = ie_df[ie_df['pval'] > 0.995]
  ie_df = ie_df[ie_df['pval'] <= 0.995]
  ie_df = ie_df.sort_values(by = 'pval')
  ie_df = ie_df.reset_index(drop = True)

  fdr_decs, hit_reject = [], False
  for idx, pval in enumerate(ie_df['pval']):
    if hit_reject:
      dec = False
    else:
      fdr_critical = ((idx + 1) / len(ie_df)) * ie_fdr_threshold
      dec = bool(pval <= fdr_critical)
    fdr_decs.append(dec)
    if dec is False and hit_reject is True:
      hit_reject = False
  ie_df['FDR accept'] = fdr_decs

  other_distribution['FDR accept'] = False
  ie_df = ie_df.append(other_distribution, ignore_index = True)
  ie_df.to_csv(out_dir + '%s_ie_dec.csv' % (treat_nm))

  print('Filtering treatment mutations explained by Illumina sequencing errors...')
  to_remove = ie_df[ie_df['FDR accept'] == False]
  adj_d = filter_illumina_error_muts(to_remove, adj_d, control_data, nm_to_seq)

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

    if 'Cas9' in treat_nm:
      continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    '''
      Empirically determined
      pickle > 37 mb: needs 4 gb ram
      pickle > 335 mb: needs 8 gb ram
    '''
    mb_file_size = _data.check_file_size(treat_nm, 'ag5a1b_subtract')
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
