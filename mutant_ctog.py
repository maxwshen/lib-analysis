# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import sklearn

# Default params
inp_dir = _config.OUT_PLACE + 'c_poswise_basic/'
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
def calc_single_stats(exp_nm, edited_threshold = 1000):
  exp_row = exp_design_df[exp_design_df['Name (new)'] == exp_nm].iloc[0]
  old_nm = exp_row['Name']

  df = pd.read_csv(inp_dir + f'{old_nm}.csv', index_col = 0)

  df['Condition'] = exp_nm
  df['Mutation'] = df['Ref nt'] + '_' + df['Obs nt']
  df['Frequency'] = df['Count'] / df['Total count']

  acc_muts = [
    'C_T',
    'C_G',
    'C_A',
  ]
  df = df[df['Mutation'].isin(acc_muts)]
  df = df.drop(columns = ['Total count', 'Ref nt', 'Obs nt'])
  df = df.pivot_table(
    index = ['Name', 'Position', 'Condition'], 
    columns = 'Mutation',
    values = 'Count', 
  ).reset_index()
  df = df.fillna(value = 0)

  df['Total edited count'] = df['C_T'] + df['C_G'] + df['C_A']
  
  df['C_A'] = df['C_A'] / df['Total edited count']
  df['C_G'] = df['C_G'] / df['Total edited count']
  df['C_T'] = df['C_T'] / df['Total edited count']

  df['C_GA'] = df['C_A'] + df['C_G']
  
  from collections import Counter
  pos_counts = Counter(df['Position'])
  keep_pos = [pos for pos in pos_counts if pos_counts[pos] >= 20]
  df = df[df['Position'].isin(keep_pos)]

  df = df[df['Total edited count'] >= edited_threshold].reset_index(drop = True)
  df = df[['Name', 'Position', 'C_GA']]
  df = df.rename(columns = {'C_GA': f'C_GA_{exp_nm}'})
  return df

##
# Merge replicates
##
def get_merged_df(nms, ddf, group_nm):
  mdf = None
  id_cols = ['Name', 'Position']
  for nm in nms:
    df = ddf[nm]
    if mdf is None:
      mdf = df
    else:
      mdf = mdf.merge(df, on = id_cols, how = 'outer')

  mdf[f'C_GA_{group_nm}'] = mdf[[f'C_GA_{nm}' for nm in nms]].apply(np.nanmean, axis = 'columns')
  return mdf


def get_lib_data(lib_nm, edited_threshold = 1000):
  mut_exp_design = exp_design_df[exp_design_df['Mutant analysis condition'].isin(['Wild-type', 'Mutant'])]

  groups = defaultdict(list)
  for idx, row in mut_exp_design.iterrows():
    editor = row['Editor (new)']
    celltype = row['Cell-type']
    library = row['Library']
    name = row['Name (new)']

    if library != lib_nm: continue

    cond = f'{celltype}_{library}_{editor}'
    groups[cond].append(name)

  print(list(groups.keys()))

  ##
  # Merge single conditions and combined replicates across editors
  # Columns are from Name (new).
  ##
  id_cols = ['Name', 'Position']
  mmdf = None
  for cond in groups:
    print(cond)

    if 'BE4' in cond: 
      edited_threshold = 1000
    elif 'eA3A' in cond and '12kChar' in cond:
      edited_threshold = 500
    elif 'eA3A' in cond and 'CtoGA' in cond:
      edited_threshold = 100
    else:
      edited_threshold = 1000

    ddf = {nm: calc_single_stats(nm, edited_threshold = edited_threshold) for nm in groups[cond]}
    mdf = get_merged_df(groups[cond], ddf, cond)

    if mmdf is None:
      mmdf = mdf
    else:
      mmdf = mmdf.merge(mdf, on = id_cols, how = 'outer')

  mmdf.to_csv(out_dir + f'mmdf_{lib_nm}.csv')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  get_lib_data('12kChar')
  get_lib_data('CtoGA')

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(exp_nm = sys.argv[1])
  # else:
  #   gen_qsubs()
  main()