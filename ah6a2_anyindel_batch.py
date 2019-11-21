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
inp_dir = _config.OUT_PLACE + 'ah6a1b_subtract/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
exp_nms = exp_design['Name']
batches = exp_design['Batch covariate (indels)']
editors = exp_design['Editor']
exp_nm_to_batch = {exp_nm: batch for exp_nm, batch in zip(exp_nms, batches)}
exp_nm_to_editor = {exp_nm: editor for exp_nm, editor in zip(exp_nms, editors)}

info_cols = ['Name', 'Category', 'Indel start', 'Indel end', 'Indel length', 'MH length', 'Inserted bases']

##
# Filters
##
def filter_mutations(to_remove, adj_d, nm_to_seq, treat_nm):
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    exp_nm = row['Name']
    mut_nm = row['MutName']

    if exp_nm not in adj_d:
      continue

    t = adj_d[exp_nm]
    cat, ids, ide, idl, mhl, ib = mut_nm.split('_')

    t_cat_set = set(t['Category'])
    if len(t_cat_set) == 1 and 'wildtype' in t_cat_set:
      continue

    crit = (t['Category'] == cat) & (t['Indel start'] == float(ids)) & (t['Indel end'] == float(ide)) & (t['Indel length'] == float(idl)) & (t['MH length'] == float(mhl)) & (t['Inserted bases'] == ib)
    t.loc[crit, 'Count'] = 0
    t = t[t['Count'] > 0]
    t['Frequency'] = t['Count'] / sum(t['Count'])

    adj_d[exp_nm] = t    
    timer.update()

  return adj_d

##
# Support
##
def form_dd(md, exp_nm):
  mut_dd = defaultdict(lambda: defaultdict(lambda: 0))
  all_mut_nms = set()

  for treat_nm in md:
    if exp_nm not in md[treat_nm]:
      continue

    df = md[treat_nm][exp_nm]

    df['Frequency'] = df['Count'] / sum(df['Count'])

    for idx, row in df.iterrows():
      if row['Category'] == 'wildtype':
        continue

      mut_nm = '%s_%s_%s_%s_%s_%s' % (
        row['Category'],
        row['Indel start'],
        row['Indel end'],
        row['Indel length'],
        row['MH length'],
        row['Inserted bases'],
      )

      mut_dd[treat_nm][mut_nm] = row['Frequency']
      all_mut_nms.add(mut_nm)

  return mut_dd, all_mut_nms

##
# Main iterator
##
def adjust_batch_effects(lib_nm):
  print(lib_nm)
  # Gather statistics
  be_treatments = []
  batch_set = set()
  batch_to_exp_nms = defaultdict(list)
  for treat_nm in treat_control_df['Treatment']:
    if 'Cas9' in treat_nm:
      continue
    if _data.get_lib_nm(treat_nm) != lib_nm:
      continue
    batch_nm = exp_nm_to_batch[treat_nm]
    be_treatments.append(treat_nm)
    batch_set.add(batch_nm)
    batch_to_exp_nms[batch_nm].append(treat_nm)

  lib_design, seq_col = _data.get_lib_design(be_treatments[0])
  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  md = dict()
  timer = util.Timer(total = len(be_treatments))
  print('Loading stats from each condition...')
  for treat_nm in be_treatments:
    with open(inp_dir + '%s.pkl' % (treat_nm), 'rb') as f:
      d = pickle.load(f)
    md[treat_nm] = d

    # df['Treatment'] = treat_nm
    # df['Batch'] = exp_nm_to_batch[treat_nm]
    # df['Editor'] = exp_nm_to_editor[treat_nm]
    timer.update()

  # ANOVA calculations
  from scipy.stats import f_oneway
  print('Calculating ANOVA on all unique indels in all target sites to identify batch effects...')

  dd = defaultdict(list)
  means_dd = defaultdict(lambda: defaultdict(lambda: dict()))
  timer = util.Timer(total = len(nms))
  for exp_nm in nms:

    mut_dd, all_mut_nms = form_dd(md, exp_nm)

    for mut_nm in all_mut_nms:
      anova_args = defaultdict(list)
      # Note: Ensure we do not implicitly treat a lack of data as an observation of zero
      for exp_nm_2 in mut_dd:
        anova_args[exp_nm_to_batch[exp_nm_2]].append(mut_dd[exp_nm_2][mut_nm])

      '''
        Ensure non-degenerate ANOVA testing.
  
        If every batch has 0 std, we have identical values. It's likely that these identical values are 0 because of the sparsity of the data when considering unique indels (highly heterogeneous) at 12,000 target sites.

        If every batch with a non-zero value has only one observation, skip. 
      '''
      # Only perform ANOVA test on indels where at least one batch has non-zero std (otherwise it was seen only once in any batch, so it's not a batch effect)
      num_non_zero_stds = 0
      mean_d, std_d = dict(), dict()
      for batch in batch_set:
        if batch in anova_args:
          mean_val = np.mean(anova_args[batch])
          std_val = np.std(anova_args[batch])
          if std_val > 0:
            num_non_zero_stds += 1
        else:
          mean_val = np.nan
          std_val = np.nan
        mean_d[batch] = mean_val
        std_d[batch] = std_val

      degenerate_flag = False
      if num_non_zero_stds == 0:
        for batch in batch_set:
          batch_data = anova_args[batch]
          if len(batch_data) == 0:
            continue
          has_non_zero = bool(batch_data.count(0) != len(batch_data))
          if has_non_zero and len(batch_data) == 1:
            degenerate_flag = True
          # elif has_non_zero and len(batch_data) > 1:
            # import code; code.interact(local=dict(globals(), **locals()))
      if degenerate_flag:
        continue

      aa = tuple([s for s in anova_args.values() if len(s) != 0])
      if len(aa) < 2:
        continue

      fstat, pval = f_oneway(*aa)
      if np.isnan(pval):
        continue
      dd['Statistic'].append(fstat)
      dd['pval'].append(pval)
      dd['MutName'].append(mut_nm)
      dd['Name'].append(exp_nm)

      for batch in batch_set:
        dd['Mean %s' % (batch)].append(mean_d[batch])
        dd['Std %s' % (batch)].append(std_d[batch])
        means_dd[exp_nm][mut_nm][batch] = mean_val

    timer.update()


  stats_df = pd.DataFrame(dd)
  if len(stats_df) == 0:
    empty_df = pd.DataFrame()
    empty_df.to_csv(out_dir + 'mutation_dec_%s.csv' % (lib_nm))
    empty_df.to_csv(out_dir + 'removed_batch_effects_%s.csv' % (lib_nm))
    empty_df.to_csv(out_dir + 'removed_stats_%s.csv' % (lib_nm))
    return
  
  stats_df['-log10p'] = -np.log10(stats_df['pval'])

  # Apply FDR
  print('Finding significant batch effects while controlling false discovery...')

  fdr_threshold = 0.01
  other_distribution = stats_df[stats_df['pval'] > 0.995]
  stats_df = stats_df[stats_df['pval'] <= 0.995]
  stats_df = stats_df.sort_values(by = 'pval')
  stats_df = stats_df.reset_index(drop = True)

  fdr_decs, hit_reject = [], False
  for idx, pval in enumerate(stats_df['pval']):
    if hit_reject:
      dec = False
    else:
      fdr_critical = ((idx + 1) / len(stats_df)) * fdr_threshold
      dec = bool(pval <= fdr_critical)
    fdr_decs.append(dec)
    if dec is False and hit_reject is True:
      hit_reject = False
  stats_df['FDR accept'] = fdr_decs

  other_distribution['FDR accept'] = False
  stats_df = stats_df.append(other_distribution, ignore_index = True)
  stats_df.to_csv(out_dir + 'mutation_dec_%s.csv' % (lib_nm))


  '''
    Identify mutations for removal
    At mutations passing Bonferroni corrected ANOVA test,
    identify batches where mutations are frequent
  '''
  print('Identifying batches to remove mutations from...')
  to_remove = stats_df[stats_df['FDR accept'] == True]

  dd = defaultdict(list)
  dd_stats = defaultdict(list)
  timer = util.Timer(total = len(to_remove))
  for idx, row in to_remove.iterrows():
    timer.update()
    exp_nm = row['Name']
    mut_nm = row['MutName']

    means = means_dd[exp_nm][mut_nm]
    mean_vals = list(means.values())
    mean_means = np.mean(mean_vals)

    for batch_nm in means:
      if means[batch_nm] >= mean_means or means[batch_nm] >= 0.005:
        dd['Batch'].append(batch_nm)
        dd['Name'].append(exp_nm)
        dd['MutName'].append(mut_nm)

    for batch_nm in means:
      dd_stats['%s' % (batch_nm)].append(means[batch_nm])
    dd_stats['MutName'].append(mut_nm)
    dd_stats['Name'].append(exp_nm)
  batch_muts_to_remove = pd.DataFrame(dd)
  batch_muts_to_remove.to_csv(out_dir + 'removed_batch_effects_%s.csv' % (lib_nm))

  batch_muts_stats = pd.DataFrame(dd_stats)
  batch_muts_stats.to_csv(out_dir + 'removed_stats_%s.csv' % (lib_nm))

  # Mutations are removed in ah6a3

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

  lib_nms = [
    '12kChar',
    'LibA',
    'AtoG',
    'CtoT',
    'CtoGA',
  ]

  num_scripts = 0
  for lib_nm in lib_nms:
    command = 'python %s.py %s' % (NAME, lib_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, lib_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=16:00:00,h_vmem=32G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
def main(lib_nm = ''):
  print(NAME)
  
  # Function calls
  adjust_batch_effects(lib_nm)

  # adjust_batch_effects('LibA')
  # adjust_batch_effects('12kChar')
  # adjust_batch_effects('AtoG')
  # adjust_batch_effects('CtoT')

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(lib_nm = sys.argv[1])
  else:
    gen_qsubs()