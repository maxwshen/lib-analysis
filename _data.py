import _config
import sys, os, pickle, subprocess
import pandas as pd

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
group_design = pd.read_csv(_config.DATA_DIR + 'master_L2_groups.csv', index_col = 0)

prj_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/'

L1_names = set(exp_design['Name'])
L2_names = set(group_design['Name'])

##
# Functions
##
def check_valid_and_fetch_level(nm):
  if nm in L1_names:
    row = exp_design[exp_design['Name'] == nm].iloc[0]
    return 'L1', row
  elif nm in L2_names:
    row = group_design[group_design['Name'] == nm].iloc[0]
    return 'L2', row
  else:
    return None, None

def load_data(nm, data_nm):
  '''
    Supported nm: An item in 
      exp_design['Name']
      group_design['Name']

    Supported data_nm: [
      'g4_poswise_be', 
      'g5_combin_be', 
      'e_newgenotype_Cas9'
      'a4_poswise_adjust',
      'a5_combin_adjust',
      'a_newgenotype_Cas9_adjust',
    ]
  '''

  level, row = check_valid_and_fetch_level(nm)
  if level is None:
    return None

  scripts_base = [
    'g4_poswise_be', 
    'g5_combin_be', 
  ]
  if data_nm in scripts_base:
    fn = prj_dir + '%s/out/%s/%s.pkl' % (row['Group'], data_nm, row['Local name'])
    with open(fn, 'rb') as f:
      data = pickle.load(f)

  if data_nm == 'e_newgenotype_Cas9':
    fn = prj_dir + '%s/out/%s/%s.csv' % (row['Group'], data_nm, row['Local name'])
    data = pd.read_csv(fn, index_col = 0)

  scripts_adjust = [
    'ag4_poswise_be_adjust', 
    'ag5_combin_be_adjust', 
    'ae_newgenotype_Cas9_adjust'
  ]
  if data_nm in scripts_adjust:
    if level == 'L1':
      fn = _config.OUT_PLACE + '%s/%s.pkl' % (data_nm, nm)
    elif level == 'L2':
      fn = _config.OUT_PLACE + 'b_group_L2/%s/%s.pkl' % (data_nm, nm)
    with open(fn, 'rb') as f:
      data = pickle.load(f)

  return data

def get_lib_design(nm):
  level, row = check_valid_and_fetch_level(nm)
  if level is None:
    return None, None

  if level == 'L1':
    l1_nm = nm
  elif level == 'L2':
    row = group_design[group_design['Name'] == nm].iloc[0]
    l1_nm = row['Item1']

  row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
  lib_nm = row['Library']

  lib_design = pd.read_csv(_config.DATA_DIR + 'library_%s.csv' % (lib_nm))
  seq_col = [s for s in lib_design.columns if 'Sequence context' in s][0]

  return lib_design, seq_col