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


##
# Helper
##
def impute_wt(df, nt_cols):
  for col in nt_cols:
    ref_nt = col[0]
    df.loc[df[col] == '.', col] = ref_nt
  return df


def is_not_wildtype(row):
  nt_cols = [col for col in row.index if col != 'Count']
  num_edits = sum([bool(row[col] != col[0]) for col in nt_cols])
  return bool(num_edits != 0)


##
# Primary
##
def compare_lib_to_endo():

  # endo_sites = sorted(list(set(endo_design['Name (no rep)'])))

  endo_sites = [
    'HEK293T_EMX_ABE-CP1041',
    'HEK293T_EMX_BE4',
    'HEK293T_EMX_BE4-CP1028',
    'HEK293T_HEK2_ABE',
    'HEK293T_HEK2_ABE-CP1041',
    # 'HEK293T_HEK3_eA3ANG',
    # 'HEK293T_HEK4_eA3A',
    # 'HEK293T_b04_eA3ANG',
    # 'HEK293T_b41_eA3ANG',
  ]

  for endo_site in endo_sites:
    print(endo_site)
    stat_dd = defaultdict(list)

    print('Loading data...')
    # try:
    data = _data.get_library_endo_datasets(endo_site, 'g5_combin_be')
    # except:
      # print(f'Failed to find a matched target site in any library for {endo_site}')
      # continue
    print('Done.')

    min_edited_count = 200

    ## Individual filtering and adjustment
    new_data = dict()
    for nm in data:
      df = data[nm]

      # Readcount filter conditions
      if sum(df['Count']) < min_edited_count:
        print(f'Filtered {nm} for insufficient edited read count, threshold {min_edited_count}')
        continue

      if 'index' in df.columns:
        df = df.drop(columns = 'index')

      nt_cols = [col for col in df.columns if col != 'Count']

      # Impute . as wt
      df = impute_wt(df, nt_cols)

      # Cleanup
      df = df.groupby(nt_cols)['Count'].agg('sum').drop_duplicates().reset_index()

      # Try to filter wild-type rows (incomplete since we drop columns later)
      df = df[df.apply(is_not_wildtype, axis = 'columns')]

      # Ensure edited readcount is high
      edited_count = sum(df['Count'])
      print(nm, edited_count)
      if edited_count < min_edited_count:
        print(f'Filtered {nm} for insufficient edited read count, threshold {min_edited_count}')
        continue

      new_data[nm] = df
    data = new_data


    ##
    # Pairwise compare
    ##
    nms = list(data.keys())
    for idx in range(len(nms)):
      for jdx in range(idx + 1, len(nms)):
        d1 = data[nms[idx]]
        d2 = data[nms[jdx]]

        shared_cols = list(set(d1.columns) & set(d2.columns))
        nt_cols = [col for col in shared_cols if col != 'Count']

        ds1 = d1.groupby(nt_cols)['Count'].agg('sum').drop_duplicates().reset_index()
        ds2 = d2.groupby(nt_cols)['Count'].agg('sum').drop_duplicates().reset_index()

        # Filter wild-type rows
        ds1 = ds1[ds1.apply(is_not_wildtype, axis = 'columns')]
        ds2 = ds2[ds2.apply(is_not_wildtype, axis = 'columns')]

        mdf = ds1.merge(ds2, on = nt_cols, how = 'outer').fillna(value = 0)


        from scipy.stats import pearsonr
        pr, pval = pearsonr(mdf['Count_x'], mdf['Count_y'])

        shared_cols_str = ','.join(sorted(list(shared_cols)))

        stat_dd['Condition 1'].append(nms[idx])
        stat_dd['Condition 2'].append(nms[jdx])
        stat_dd['Shared columns'].append(shared_cols_str)
        stat_dd['Pearsonr'].append(pr)

        try:
          mdf.to_csv(out_dir + f"{endo_site}_{nms[idx]}_{nms[jdx]}_{pr:.2f}.csv")
        except OSError:
          print('Failed to save CSV, file name too long?')
          continue

        print(f"{endo_site}, {nms[idx]}, {nms[jdx]}: pearsonr {pr:.2f}")

    stat_df = pd.DataFrame(stat_dd)
    if len(stat_df) == 0: 
      print('No pairs found for comparison')
      continue
    stat_df.to_csv(out_dir + f'_{endo_site}.csv')

    pv_df = stat_df.pivot(index = 'Condition 1', columns = 'Condition 2', values = 'Pearsonr')
    pv_df.to_csv(out_dir + f'_{endo_site}_pivot.csv')

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
    treat_nm = row['Treatment']
    if 'Cas9' in treat_nm:
      continue
    lib_nm = _data.get_lib_nm(treat_nm)
    if lib_nm == 'LibA':
      num_targets = 2000
      num_targets_per_split = 200
    elif lib_nm == 'CtoGA':
      num_targets = 4000
      num_targets_per_split = 500
    else:
      num_targets = 12000
      num_targets_per_split = 2000

    for start_idx in range(0, num_targets, num_targets_per_split):
      end_idx = start_idx + num_targets_per_split - 1

      # Skip completed
      # out_pkl_fn = out_dir + '%s_%s_%s.pkl' % (treat_nm, start_idx, end_idx)
      # if os.path.isfile(out_pkl_fn):
      #   if os.path.getsize(out_pkl_fn) > 0:
      #     continue

      command = 'python %s.py %s %s %s' % (NAME, treat_nm, start_idx, end_idx)
      script_id = NAME.split('_')[0]

      try:
        mb_file_size = _data.check_file_size(treat_nm, 'ag5a4_profile_subset')
      except FileNotFoundError:
        mb_file_size = 0
      ram_gb = 2
      if mb_file_size > 140:
        ram_gb = 4
      if mb_file_size > 400:
        ram_gb = 8
      if mb_file_size > 1000:
        ram_gb = 16

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, start_idx)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      num_scripts += 1

      # Write qsub commands
      qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=%sG -wd %s %s &' % (ram_gb, _config.SRC_DIR, sh_fn))

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
  
  # Function calls
  compare_lib_to_endo()

  return

'''
  Note (May 24th, 2019):
  Runtime for just 2 sites is quite long, about 5 minutes. Definitely consider qsub'ing in the future.
'''
if __name__ == '__main__':
  if len(sys.argv) > 1:
    main()
  else:
    # gen_qsubs()
    main()