# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

# Default params
inp_dir = _config.OUT_PLACE + 'endo_fulledits_stats/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

endo_design = pd.read_csv(_config.DATA_DIR + 'endo_design.csv', index_col = 0)
amplicon_design = pd.read_csv(_config.DATA_DIR + 'endo_amplicon_design.csv')

##
#
##
def is_not_wildtype(row):
  nt_cols = [col for col in row.index if col != 'Count']
  num_edits = sum([bool(row[col] != col[0]) for col in nt_cols])
  return bool(num_edits != 0)


##
# Primary
##
def calc_stats(nm, design_row):
  stats = dict()
  try:
    df = pd.read_csv(inp_dir + f'g5_{nm}.csv', index_col = 0)
  except:
    return None
  df = df[df.apply(is_not_wildtype, axis = 'columns')]
  stats['Condition name'] = nm
  stats['Total edited reads'] = sum(df['Count'])

  ctvs = list('GA')

  snp_cols = [f'C{s}' for s in range(1, 10)]
  for snp_col in snp_cols:
    if snp_col not in df.columns: continue

    denom = sum(df[df[snp_col] != 'C']['Count'])
    stats[f'{snp_col} transversion purity'] = sum(df[df[snp_col].isin(ctvs)]['Count']) / denom if denom > 0 else np.nan

    denom = sum(df['Count'])
    stats[f'{snp_col} editing efficiency'] = sum(df[df[snp_col] != 'C']['Count']) / denom if denom > 0 else np.nan

  return stats


##
# qsub
##
def gen_qsubs():
  # # Generate qsub shell scripts and commands for easy parallelization
  # print('Generating qsub scripts...')
  # qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  # util.ensure_dir_exists(qsubs_dir)
  # qsub_commands = []

  # # Generate qsubs only for unfinished jobs
  # treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  # num_scripts = 0
  # for idx, row in treat_control_df.iterrows():
  #   treat_nm = row['Treatment']
  #   if 'Cas9' in treat_nm:
  #     continue
  #   lib_nm = _data.get_lib_nm(treat_nm)
  #   if lib_nm == 'LibA':
  #     num_targets = 2000
  #     num_targets_per_split = 200
  #   elif lib_nm == 'CtoGA':
  #     num_targets = 4000
  #     num_targets_per_split = 500
  #   else:
  #     num_targets = 12000
  #     num_targets_per_split = 2000

  #   for start_idx in range(0, num_targets, num_targets_per_split):
  #     end_idx = start_idx + num_targets_per_split - 1

  #     # Skip completed
  #     # out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx)
  #     # if os.path.isfile(out_pkl_fn):
  #     #   if os.path.getsize(out_pkl_fn) > 0:
  #     #     continue

  #     command = 'python %s.py %s %s %s' % (NAME, treat_nm, start_idx, end_idx)
  #     script_id = NAME.split('_')[0]

  #     try:
  #       mb_file_size = _data.check_file_size(treat_nm, 'ag5a4_profile_subset')
  #     except FileNotFoundError:
  #       mb_file_size = 0
  #     ram_gb = 2
  #     if mb_file_size > 140:
  #       ram_gb = 4
  #     if mb_file_size > 400:
  #       ram_gb = 8
  #     if mb_file_size > 1000:
  #       ram_gb = 16

  #     # Write shell scripts
  #     sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, start_idx)
  #     with open(sh_fn, 'w') as f:
  #       f.write('#!/bin/bash\n%s\n' % (command))
  #     num_scripts += 1

  #     # Write qsub commands
  #     qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=%sG -wd %s %s &' % (ram_gb, _config.SRC_DIR, sh_fn))

  # # Save commands
  # commands_fn = qsubs_dir + '_commands.sh'
  # with open(commands_fn, 'w') as f:
  #   f.write('\n'.join(qsub_commands))

  # subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  # print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return

##
# Main
##
@util.time_dec
def main():
  '''
    Input: g5 from endo_fulledits_stats.

    Calculates cytosine transversion purity by position.
  '''
  print(NAME)
  
  endo_sites = [
    'HEK2',
    'HEK3',
    'HEK4',
    'RNF2',
  ]

  adf = amplicon_design[amplicon_design['Sequence name'].isin(endo_sites)]

  # Function calls
  timer = util.Timer(total = len(adf))
  for idx, design_row in adf.iterrows():
    endo_loci_nm = design_row['Sequence name']
    dfs = endo_design[endo_design['Endogenous loci name'] == endo_loci_nm]

    dd = defaultdict(list)
    for nm in dfs['Name']:
      if 'ABE' in nm: continue
      stats = calc_stats(nm, design_row)
      if stats is None:
        continue
      for col in stats:
        dd[col].append(stats[col])
      for col in ['Sequence name', 'gRNA']:
        dd[col].append(design_row[col])

    df = pd.DataFrame(dd)
    df.to_csv(out_dir + f'{endo_loci_nm}.csv')

    timer.update()

  return

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main()
  else:
    # gen_qsubs()
    main()