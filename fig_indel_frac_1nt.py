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
inp_dir_lib_nocorrect = _config.OUT_PLACE + 'indel_anyindel_pos/'
inp_dir_lib_corrected = _config.OUT_PLACE + 'indel_anyindel_pos_c1bpfq_v2/'
inp_dir_endo = _config.OUT_PLACE + 'endo_fulledits_stats/'

NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
endo_design = pd.read_csv(_config.DATA_DIR + 'endo_design.csv', index_col = 0)


##
# Form groups
##
def get_library_stats(inp_dir_lib, data_type):
  import glob
  fns = glob.glob(inp_dir_lib + f'*len_melt.csv')
  
  dd = defaultdict(list)
  timer = util.Timer(total = len(fns))
  for fn in fns:
    cond_nm = fn.split('/')[-1].replace('_len_melt.csv', '')
    df = pd.read_csv(fn, index_col = 0)

    dd['Name'].append(cond_nm)
    dd['Data type'].append(data_type)
    dd['Editor'].append(exp_design_df[exp_design_df['Name'] == cond_nm]['Editor'].iloc[0])

    fq_1nt_indels = sum(df[df['Indel length'].isin([-1, 1])]['Frequency']) / sum(df['Frequency'])
    dd['Fraction of 1-bp indels in indels'].append(fq_1nt_indels)

    fq_1nt_dels = sum(df[df['Indel length'].isin([-1])]['Frequency']) / sum(df['Frequency'])
    dd['Fraction of 1-bp dels in indels'].append(fq_1nt_dels)

    fq_1nt_ins = sum(df[df['Indel length'].isin([-1])]['Frequency']) / sum(df['Frequency'])
    dd['Fraction of 1-bp ins in indels'].append(fq_1nt_ins)

    timer.update()

  return pd.DataFrame(dd)

def get_endo_stats():
  df = pd.read_csv(inp_dir_endo + f'_all_stats.csv', index_col = 0)

  stat_cols = [
    'Indel read count',
    'Base edited read count',
    'Total read count',
    'Indel frequency in all reads',
    'BE:indel ratio',
    'Fraction of 1-bp dels in indels',
    'Fraction of 1-bp ins in indels',
    'Fraction of 1-bp indels in indels',
  ]

  dd = defaultdict(list)
  for idx, row in df.iterrows():
    if row['Indel read count'] <= 200:
      continue
    if row['Total read count'] <= 1000:
      continue

    dd['Name'].append(row['Name'])
    dd['Editor'].append(endo_design[endo_design['Name'] == row['Name']]['Editor'].iloc[0])
    dd['Data type'].append('Endogenous')
    print(row['Name'])
    for col in stat_cols:
      dd[col].append(row[col])

  stats_df = pd.DataFrame(dd)
  stats_df.to_csv(out_dir + 'endo_stats.csv')

  return stats_df

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
  all_conds = treatments + controls

  num_scripts = 0
  for cond in all_conds:

    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

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
def main():
  print(NAME)

  endo_df = get_endo_stats()
  lib_df = get_library_stats(inp_dir_lib_nocorrect, 'Library - without 1-nt indel fq. correction')
  lib_df2 = get_library_stats(inp_dir_lib_corrected, 'Library - with 1-nt indel fq. correction')

  mdf = endo_df.append(lib_df, ignore_index = True, sort = False)
  mdf = mdf.append(lib_df2, ignore_index = True, sort = False)

  mdf.to_csv(out_dir + 'fig_indel_frac_1nt-stats.csv')


  return


if __name__ == '__main__':
  main()