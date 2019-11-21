# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle, copy
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import sklearn

# Default params
inp_dir_indel = _config.OUT_PLACE + 'indel_anyindel_pos/'
inp_dir_ah6c = _config.OUT_PLACE + 'ah6c_reduce_1bp_indel_fq/'
inp_dir_fulledits = _config.OUT_PLACE + 'fulledits_ratio2_correct1bpfq/'
inp_dir_anova = _config.OUT_PLACE + 'fig_beindelratio_anova_c1bpfq/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)


##
# Primary logic
##
def indel_anyindel_pos(exp_nm):
  '''
    Read in position-adjusted indel CSV from indel_anyindel_pos. Then:
    1. Apply higher readcount filter than 100 reads per target site, from fulledits_ratio2_correct1bpfq.
    2. Reduce 1-bp indel freq. by reduce factor, from ah6c_reduce_1bp_indel_fq.
    3. Batch-adjust indel counts and frequencies.
  '''

  editor = exp_design_df[exp_design_df['Name'] == exp_nm]['Editor'].iloc[0]

  # reduce_stat_df = pd.read_csv(inp_dir_ah6c + f'stats_{exp_nm}.csv', index_col = 0)
  reduce_stat_df = pd.read_csv(inp_dir_ah6c + f'stats_reduce_factors.csv', index_col = 0)
  if editor != 'UT':
    reduce_factor = float(reduce_stat_df[reduce_stat_df['Condition'] == exp_nm]['Reduce factor'].iloc[0])
  else:
    reduce_factor = 1

  # Get np.log scale batch adjustment factor
  if editor != 'UT':
    anova_df = pd.read_csv(inp_dir_anova + f'linreg_coefs.csv', index_col = 0)
    batch_df = anova_df[anova_df['Feature type'] == 'Batch']
    batch_nm = 'x0_' + exp_design_df[exp_design_df['Name'] == exp_nm]['Batch covariate'].iloc[0]
    mean_batch = np.mean(batch_df['Coefficient'])
    batch_adjust = batch_df[batch_df['Feature name'] == batch_nm]['Coefficient'].iloc[0] - mean_batch
  else:
    batch_adjust = 0

  # Adjust mdf
  mdf = pd.read_csv(inp_dir_indel + f'{exp_nm}.csv', index_col = 0)
  mdf.loc[mdf['Indel length'] == 1, 'Count'] *= reduce_factor
  if editor != 'UT':
    ok_target_sites_df = pd.read_csv(inp_dir_fulledits + f'{exp_nm}.csv', index_col = 0)
    ok_target_sites = set(ok_target_sites_df['Name (unique)'])
  else:
    ok_target_sites = set(mdf['Name'])
  mdf = mdf[mdf['Name'].isin(ok_target_sites)]
  mdf['Count'] = np.exp(np.log(mdf['Count']) + batch_adjust)
  mdf['Frequency'] = np.exp(np.log(mdf['Frequency']) + batch_adjust)
  mdf.to_csv(out_dir + f'{exp_nm}.csv')

  # Adjust pos_melt by nt
  indel_lens = range(-20, 15 + 1)
  timer = util.Timer(total = len(indel_lens))
  for indel_len in indel_lens:
    inp_fn = inp_dir_indel + f'{exp_nm}_pos_{indel_len}nt.csv'
    if not os.path.isfile(inp_fn):
      continue
    df = pd.read_csv(inp_fn, index_col = 0)
    df = df[df['Name'].isin(ok_target_sites)]
    for col in [col for col in df.columns if col != 'Name']:
      if abs(indel_len) == 1:
        df[col] *= reduce_factor
      # Issue: log of 0
      df.loc[df[col] > 0, col] = np.exp(np.log(df.loc[df[col] > 0, col]) + batch_adjust)

    df.to_csv(out_dir + f'{exp_nm}_pos_{indel_len}nt.csv')
    timer.update()

  # Adjust len_melt
  lm_df = pd.read_csv(inp_dir_indel + f'{exp_nm}_len_melt.csv', index_col = 0)
  lm_df = lm_df[lm_df['Name'].isin(ok_target_sites)]
  lm_df.loc[lm_df['Indel length'].isin([1, -1]), 'Frequency'] *= reduce_factor
  lm_df['Frequency'] = np.exp(np.log(lm_df['Frequency']) + batch_adjust)
  lm_df.to_csv(out_dir + f'{exp_nm}_len_melt.csv')

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

  treatments = list(set(treat_control_df['Treatment']))
  controls = list(set(treat_control_df['Control']))
  # all_conds = treatments + controls
  # all_conds = controls
  all_conds = treatments

  num_scripts = 0
  for cond in all_conds:
    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

    if os.path.isfile(out_dir + f'{cond}.csv'):
      continue

    if 'Cas9' in cond:
      continue

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, cond)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=6:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
def main(exp_nm = ''):
  print(NAME)
  
  if exp_nm == 'run_all':
    treatments = list(set(treat_control_df['Treatment']))
    for exp_nm in sorted(treatments):
      print(exp_nm)
      try:
        indel_anyindel_pos(exp_nm)
      except:
        print('Failed')
  else:
    indel_anyindel_pos(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()