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

def gather_stats_binom_control_muts(t, c, seq, treat_nm, nm, decisions):
  '''
    Filter treatment mutations that can be explained by control freq.
    In practice, this step is most effective for control mutations
    with relatively high frequency => relatively high variance

    Considers all events that occur (fq > 0%) in both control and treatment data
  '''
  fpr_threshold_try1 = 0.10
  for jdx, ref_nt in enumerate(seq):
    c_tot = sum(c[jdx])
    t_tot = sum(t[jdx])
    for kdx in range(len(t[jdx])):
      if kdx == nt_to_idx[ref_nt] or t[jdx][kdx] == 0:
        continue

      c_fq = c[jdx][kdx] / c_tot
      t_fq = t[jdx][kdx] / t_tot
      pval = binom.sf(t[jdx][kdx] - 1, t_tot, c_fq)

      if c_fq > 0:
        decisions['obs_nt'].append(nts[kdx])
        decisions['ref_nt'].append(ref_nt)
        decisions['c_fq'].append(c_fq)
        decisions['c_ct'].append(c[jdx][kdx])
        decisions['t_fq'].append(t_fq)
        decisions['t_ct'].append(t[jdx][kdx])
        decisions['c_tot'].append(c_tot)
        decisions['t_tot'].append(t_tot)
        decisions['idx'].append(jdx)
        decisions['pos'].append(_data.idx_to_pos(jdx, treat_nm))
        decisions['pval'].append(pval)
        decisions['nm'].append(nm)

  return

def gather_stats_illumina_errors(t, c, t_minq, c_minq, seq, treat_nm, nm, decisions):
  '''
    Identify mutations explainable by Illumina sequencing error
    Filtered at Q30 (1e-3), most columns have minimum
    Q = 32 (6e-4), or
    Q = 36 (2e-4)

    Considers all events (>0 freq.) in treatment data.
  '''
  fpr_threshold = 0.05
  for jdx, ref_nt in enumerate(seq):
    t_tot = np.sum(t[jdx])

    t_bin_p = 10**(-t_minq[jdx] / 10) 
    c_bin_p = 10**(-c_minq[jdx] / 10)

    for kdx in range(len(t[jdx])):
      if kdx == nt_to_idx[ref_nt]:
        continue

      if t[jdx][kdx] > 0:
        t_fq = t[jdx][kdx] / t_tot
        pval = binom_minus_binom_pval(t[jdx][kdx], t_bin_p, c_bin_p, t_tot)

        decisions['obs_nt'].append(nts[kdx])
        decisions['ref_nt'].append(ref_nt)
        decisions['t_bin_p'].append(t_bin_p)
        decisions['c_bin_p'].append(c_bin_p)
        decisions['t_ct'].append(t[jdx][kdx])
        decisions['t_fq'].append(t_fq)
        decisions['t_tot'].append(t_tot)
        decisions['idx'].append(jdx)
        decisions['pos'].append(_data.idx_to_pos(jdx, treat_nm))
        decisions['pval'].append(pval)
        decisions['nm'].append(nm)
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

  for jdx, ref_nt in enumerate(seq):
    c_tot = sum(c[jdx])
    t_tot = sum(t[jdx])
    
    wipe_col = False
    for kdx in range(len(t[jdx])):
      if kdx == nt_to_idx[ref_nt] or t[jdx][kdx] == 0:
        continue

      c_fq = c[jdx][kdx] / c_tot
      t_fq = t[jdx][kdx] / t_tot

      if c_fq > 0:
        decisions['obs_nt'].append(nts[kdx])
        decisions['ref_nt'].append(ref_nt)
        decisions['c_fq'].append(c_fq)
        decisions['c_tot'].append(c_tot)
        decisions['idx'].append(jdx)
        decisions['pos'].append(_data.idx_to_pos(jdx, treat_nm))
        decisions['nm'].append(nm)

        if c_fq >= max_control_mut_fq:
          wipe_col = True
          decisions['wiped'].append(True)
        else:
          decisions['wiped'].append(False)

    if wipe_col:
      for kdx in range(len(t[jdx])):
        t[jdx][kdx] = 0
  return t

def subtract_treatment_control(t, c, seq):
  # Rescale readcounts in control to treatment's scale
  c /= np.sum(c, axis = 1, keepdims = True)
  np.nan_to_num(c, 0)
  c *= np.sum(t, axis = 1, keepdims = True)

  # Zero out wild-type elements
  for jdx, ref_nt in enumerate(seq):
    c[jdx][nt_to_idx[ref_nt]] = 0

  # Adjust treatment by control
  t -= c

  # Set negative values to zero
  t = t.clip(0)

  return t

def filter_binom_control_muts(to_remove, adj_d, control_data, nm_to_seq):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    nm = row['nm']

    t = adj_d[nm]
    c = control_data[nm]
    seq = nm_to_seq[nm]

    jdx = row['idx']
    kdx = nt_to_idx[row['obs_nt']]

    t[jdx][kdx] = 0
    adj_d[nm] = t
    timer.update()

  return adj_d

def filter_illumina_error_muts(to_remove, adj_d, control_data, nm_to_seq):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    nm = row['nm']

    t = adj_d[nm]
    c = control_data[nm]
    seq = nm_to_seq[nm]

    jdx = row['idx']
    kdx = nt_to_idx[row['obs_nt']]
    ref_nt = seq[jdx]

    t[jdx][nt_to_idx[ref_nt]] += t[jdx][kdx]
    t[jdx][kdx] = 0
    adj_d[nm] = t
    timer.update()

  return adj_d

##
# Main iterator
##
def adjust_treatment_control(treat_nm, control_nm):
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''

  treat_data = _data.load_data(treat_nm, 'g4_poswise_be')
  control_data = _data.load_data(control_nm, 'g4_poswise_be')

  treat_minq = _data.load_minq(treat_nm, 'g4_poswise_be')
  control_minq = _data.load_minq(control_nm, 'g4_poswise_be')

  lib_design, seq_col = _data.get_lib_design(treat_nm)

  adj_d = dict()
  stats_dd = defaultdict(list)

  hc_decisions = defaultdict(list)

  nm_to_seq = dict()

  '''
    Filter positions with abnormally high control mut freq. 
  '''
  print('Filtering positions with high frequency control mutations...')
  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    seq = row[seq_col]
    nm_to_seq[nm] = seq
    timer.update()

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
    adj_d[nm] = t

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + '%s_stats.csv' % (treat_nm))

  hc_df = pd.DataFrame(hc_decisions)
  hc_df = hc_df.sort_values(by = 'c_fq', ascending = False)
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


  '''
  '''
  print('Subtracting control from treatment data...')
  timer = util.Timer(total = len(adj_d))
  for nm in adj_d:
    t = adj_d[nm]
    c = control_data[nm]
    seq = nm_to_seq[nm]

    t = subtract_treatment_control(t, c, seq)
    adj_d[nm] = t
    timer.update()

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

  mut_summary = ie_df[ie_df['FDR accept'] == True]
  mut_summary['Frequency'] = mut_summary['t_fq']
  mut_summary['Count'] = mut_summary['t_ct']
  mut_summary['Total count'] = mut_summary['t_tot']
  mut_summary = mut_summary.drop(columns = ['t_bin_p', 'c_bin_p', 'pval', 'idx', 't_fq', 't_ct', 't_tot', 'FDR accept'])
  mut_summary.to_csv(out_dir + '%s_summary.csv' % (treat_nm))

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
      continue
    if 'Cas9' in treat_nm:
      continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, control_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -l h_rt=2:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  if treat_nm == '' and control_nm == '':

    treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

    for idx, row in treat_control_df.iterrows():
      treat_nm, control_nm = row['Treatment'], row['Control']
      if 'Cas9' in treat_nm:
        continue
      print(treat_nm, control_nm)
      if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
        print('... already complete')
      else:
        adjust_treatment_control(treat_nm, control_nm)

  else:
    adjust_treatment_control(treat_nm, control_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1], control_nm = sys.argv[2])
  else:
    gen_qsubs()
    # main()