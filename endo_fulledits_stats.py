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

endo_design = pd.read_csv(_config.DATA_DIR + 'endo_design.csv', index_col = 0)
profile_design = pd.read_csv(_config.DATA_DIR + 'editor_profiles.csv')


##
# Form g5 combinatorial df
##
def impute_wt(df, nt_cols):
  for col in nt_cols:
    ref_nt = col[0]
    df.loc[df[col] == '.', col] = ref_nt
  return df


def get_editor_mutations(editor_nm):
  try:
    row = profile_design[profile_design['Editor name'] == editor_nm].iloc[0]
  except:
    print(f'{editor_nm} not in editor_profiles.csv : Defaulting to BE4')
    row = profile_design[profile_design['Editor name'] == 'BE4'].iloc[0]

  muts_dict = defaultdict(list)
  mut_cols = [col for col in profile_design if col != 'Editor name']
  for col in mut_cols:
    if type(row[col]) != str:
      continue

    [ref_nt, obs_nt] = col.replace(' to ', ' ').split()
    [start_pos, end_pos] = row[col].replace('(', '').replace(')', '').replace(',', '').split()
    start_pos, end_pos = int(start_pos), int(end_pos)

    for pos in range(start_pos, end_pos + 1):
      mut_col_nm = '%s%s' % (ref_nt, pos)
      muts_dict[mut_col_nm].append(obs_nt)
  return muts_dict


##
# Statistics
##
def is_not_wildtype(row):
  nt_cols = [col for col in row.index if col != 'Count']
  num_edits = sum([bool(row[col] != col[0]) for col in nt_cols])
  return bool(num_edits != 0)

def has_ctog(row):
  nt_cols = [col for col in row.index if col != 'Count']
  for nt_col in nt_cols:
    if nt_col[0] == 'C' and row[nt_col] == 'G':
      return True
  return False

def has_ctoa(row):
  nt_cols = [col for col in row.index if col != 'Count']
  for nt_col in nt_cols:
    if nt_col[0] == 'C' and row[nt_col] == 'A':
      return True
  return False

def has_ctoga(row):
  nt_cols = [col for col in row.index if col != 'Count']
  for nt_col in nt_cols:
    if nt_col[0] == 'C' and row[nt_col] in list('GA'):
      return True
  return False

def has_ctot(row):
  nt_cols = [col for col in row.index if col != 'Count']
  for nt_col in nt_cols:
    if nt_col[0] == 'C' and row[nt_col] == 'T':
      return True
  return False




##
# Primary
##
def calc_stats(nm):
  '''
    Base editing summary statistics:
    - Total number of reads
    - Base edited read count
    - Indel read count
    - Wild-type read count

    - BE editing fraction
    - BE+indel editing fraction
    - Indel editing fraction
    - BE:indel ratio
    - Fraction of indels that are 1-bp deletions
    - Fraction of indels that are 1-bp insertions
    - Fraction of indels that are 1-bp indels

    - CtoT editing fraction out of all reads
    - CtoGA editing fraction out of all reads
    - CtoG editing fraction out of all reads
    - CtoA editing fraction out of all reads

    - CtoT editing fraction among base edited reads
    - CtoGA editing fraction among base edited reads
    - CtoG editing fraction among base edited reads
    - CtoA editing fraction among base edited reads
    
  '''
  stats = dict()
  stats['Name'] = nm
  try:
    g5d = _data.load_data(nm, 'g5_combin_be')

    # Impute . as wt
    nt_cols = [col for col in g5d.columns if col != 'Count']
    g5d = impute_wt(g5d, nt_cols)
    g5d = g5d.groupby(nt_cols)['Count'].agg('sum').drop_duplicates().reset_index()

    # Profile subset
    editor_nm = endo_design[endo_design['Name'] == nm]['Editor'].iloc[0]
    muts_dict = get_editor_mutations(editor_nm)
    editor_mut_set = set(muts_dict.keys())
    _mut_cols = [col for col in g5d.columns if col in editor_mut_set]
    g5d = g5d[_mut_cols + ['Count']]

    g5d.to_csv(out_dir + f'g5_{nm}.csv')

    # Subset to edited rows
    base_edited_df = g5d[g5d.apply(is_not_wildtype, axis = 'columns')]

    h6d = _data.load_data(nm, 'h6_anyindel')

  except FileNotFoundError:
    print(f'WARNING: Could not load data for {nm}')
    return pd.DataFrame(stats, index = [0])

  indel_total = sum(h6d[h6d['Category'].isin(['del', 'ins'])]['Count'])
  base_edited_total = sum(base_edited_df['Count'])
  reads_total = sum(h6d['Count'])
  stats['Indel read count'] = indel_total
  stats['Base edited read count'] = base_edited_total
  stats['Total read count'] = reads_total
  stats['Wild-type read count'] = reads_total - base_edited_total - indel_total

  stats['Indel frequency in all reads'] = indel_total / reads_total if reads_total > 0 else np.nan
  stats['Base editing frequency in all reads'] = base_edited_total / reads_total if reads_total > 0 else np.nan
  stats['BE+indel frequency in all reads'] = (base_edited_total + indel_total) / reads_total if reads_total > 0 else np.nan
  stats['BE:indel ratio'] = base_edited_total / indel_total if indel_total > 0 else np.nan

  crit = (h6d['Category'] == 'del') & (h6d['Indel length'] == 1)
  stats['Fraction of 1-bp dels in indels'] = sum(h6d[crit]['Count']) / indel_total if indel_total > 0 else np.nan
  crit = (h6d['Category'] == 'ins') & (h6d['Indel length'] == 1)
  stats['Fraction of 1-bp ins in indels'] = sum(h6d[crit]['Count']) / indel_total if indel_total > 0 else np.nan
  crit = (h6d['Indel length'] == 1)
  stats['Fraction of 1-bp indels in indels'] = sum(h6d[crit]['Count']) / indel_total if indel_total > 0 else np.nan

  ctog_count = sum(base_edited_df[base_edited_df.apply(has_ctog, axis = 'columns')]['Count']) if len(base_edited_df) > 0 else np.nan
  ctoa_count = sum(base_edited_df[base_edited_df.apply(has_ctoa, axis = 'columns')]['Count']) if len(base_edited_df) > 0 else np.nan
  ctoga_count = sum(base_edited_df[base_edited_df.apply(has_ctoga, axis = 'columns')]['Count']) if len(base_edited_df) > 0 else np.nan
  ctot_count = sum(base_edited_df[base_edited_df.apply(has_ctot, axis = 'columns')]['Count']) if len(base_edited_df) > 0 else np.nan

  stats['Fraction of base edited reads with >1 CtoGA edit'] = ctoga_count / base_edited_total if base_edited_total > 0 else np.nan
  stats['Fraction of base edited reads with >1 CtoG edit'] = ctog_count / base_edited_total if base_edited_total > 0 else np.nan
  stats['Fraction of base edited reads with >1 CtoA edit'] = ctoa_count / base_edited_total if base_edited_total > 0 else np.nan
  stats['Fraction of base edited reads with >1 CtoT edit'] = ctot_count / base_edited_total if base_edited_total > 0 else np.nan

  stats['Fraction of all reads with >1 CtoGA edit'] = ctoga_count / reads_total if reads_total > 0 else np.nan
  stats['Fraction of all reads with >1 CtoG edit'] = ctog_count / reads_total if reads_total > 0 else np.nan
  stats['Fraction of all reads with >1 CtoA edit'] = ctoa_count / reads_total if reads_total > 0 else np.nan
  stats['Fraction of all reads with >1 CtoT edit'] = ctot_count / reads_total if reads_total > 0 else np.nan

  stats_df = pd.DataFrame(stats, index = [0])  

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

  # Generate qsubs only for unfinished jobs
  treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  num_scripts = 0
  for nm in endo_design['Name']:

    out_fn = out_dir + f'{nm}.csv'

    if '190903' not in nm:
      if os.path.isfile(out_fn): continue

    # if os.path.isfile(out_fn): continue

    command = f'python {NAME}.py {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{nm}.sh'
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=1G -wd {_config.SRC_DIR} {sh_fn} &')

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
def main(argv):
  print(NAME)
  
  nm = argv[0]

  if nm != '_combine':
    df = calc_stats(nm)
    df.to_csv(out_dir + f'{nm}.csv')
  else:
    mdf = pd.DataFrame()
    timer = util.Timer(total = len(endo_design))
    for nm in endo_design['Name']:
      df = pd.read_csv(out_dir + f'{nm}.csv', index_col = 0)
      mdf = mdf.append(df, ignore_index = True, sort = False)
      timer.update()
    mdf.to_csv(out_dir + '_all_stats.csv')

  return

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
