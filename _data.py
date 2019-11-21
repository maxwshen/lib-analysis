import _config
import sys, os, pickle, subprocess
import pandas as pd, numpy as np

exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
group_design = pd.read_csv(_config.DATA_DIR + 'master_L2_groups.csv', index_col = 0)
endo_design = pd.read_csv(_config.DATA_DIR + 'endo_design.csv', index_col = 0)
endo_to_library = pd.read_csv(_config.DATA_DIR + 'endo_to_library.csv', index_col = 0)
treatment_control_design = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

prj_dir = '/ahg/regevdata/projects/CRISPR-libraries/prj/'

lib_dfs = dict()
for lib_nm in ['LibA', '12kChar', 'CtoT', 'AtoG', 'CtoGA', 'PAMvar']:
  lib_dfs[lib_nm] = pd.read_csv(_config.DATA_DIR + f'library_{lib_nm}.csv', index_col = 0)

L1_names = set(exp_design['Name'])
L2_names = set(group_design['Name'])
endo_names = set(endo_design['Name'])

# Train models for these editors
main_base_editors = [
  'ABE',
  'ABE-CP1040',
  'AID',
  'BE4',
  'BE4-CP1028',
  'BE4max_H47ES48A',
  'BE4_H47ES48A',
  'CDA',
  'eA3A',
  'evoAPOBEC',
  'eA3A_T31AT44A',
  'eA3A_T31A',
  'eA3A_noUGI',
  'eA3A_Nprime_UGId12_T30A',
  'eA3Amax_T44DS45A',
  'eA3A_noUGI_T31A',
]

editor_renaming = {
  'ABE': 'ABE',
  'ABE8': 'ABE8',
  'ABE-CP1044': 'ABE-CP',
  'ABE-CP1040': 'ABE-CP',
  'BE4': 'BE4',
  'BE4-CP1028': 'BE4-CP',
  'AID': 'AID-BE4',
  'BE4max_H47ES48A': 'BE4 H47ES48A',
  'BE4_H47ES48A': 'BE4 H47ES48A',
  'CDA': 'CDA-BE4',
  'eA3A': 'eA3A-BE4',
  'evoAPOBEC': 'evoA-BE4',
  'evoA': 'evoA-BE4',
  'eA3A_T31AT44A': 'eA3A-BE4 T31AT44A',
  'eA3A_T31A': 'eA3A-BE4, T31A',
  'eA3A_noUGI': 'eA3A_noUGI',
  'eA3A_Nprime_UGId12_T30A': 'eA3A_Nprime_UGId12_T30A',
  'eA3Amax_T44DS45A': 'eA3A-BE4 T44DS45A',
  'eA3A_noUGI_T31A': 'eA3A_noUGI_T31A',
}

##
# Helper functions
##
def check_valid_and_fetch_level(nm):
  if nm in L1_names:
    row = exp_design[exp_design['Name'] == nm].iloc[0]
    return 'L1', row
  elif nm in L2_names:
    row = group_design[group_design['Name'] == nm].iloc[0]
    return 'L2', row
  elif nm in endo_names:
    row = endo_design[endo_design['Name'] == nm].iloc[0]
    return 'endo', row
  else:
    # print('Bad name: %s' % (nm))
    raise ValueError
    return None, None

##
# Data fetching
##
def check_file_size(nm, data_nm):
  level, row = check_valid_and_fetch_level(nm)
  if level is None:
    return None

  scripts_base = [
    'g4_poswise_be', 
    'g5_combin_be', 
    'h4_poswise_del',
    'h6_anyindel',
  ]
  if data_nm in scripts_base:
    fn = prj_dir + '%s/out/%s/%s.pkl' % (row['Group'], data_nm, row['Local name'])

  if data_nm in ['e_newgenotype_Cas9']:
    fn = prj_dir + '%s/out/%s/%s.csv' % (row['Group'], data_nm, row['Local name'])


  scripts_adjust = [
    'ag4_poswise_be_adjust', 
    'ag5_combin_be_adjust', 
    'ae_newgenotype_Cas9_adjust',
    'ag4b_adjust_background',
    # 'ag4e_find_highfreq_batcheffects',
    'ag4a2_adjust_batch_effects',
    'ag4b4_filter_inprofile_batch_effects',
    'ag5a1a_hf_bc',
    'ag5a1b_subtract',
    'ag5a1c_ie',
    'ag5a2_adjust_batch_effects',
    'ag5a3_adjust_inprofile_batch_effects',
    'ag5a4_profile_subset',
    'ah6a1a_hf_bc',
    'ah6a1b_subtract',
    'ah6a3_remove_batch',
    'ah6c_reduce_1bp_indel_fq',
  ]
  if data_nm in scripts_adjust:
    if level == 'L1':
      fn = _config.OUT_PLACE + '%s/%s.pkl' % (data_nm, nm)
    elif level == 'L2':
      fn = _config.OUT_PLACE + 'b_group_L2/%s/%s.pkl' % (data_nm, nm)

  file_size = os.path.getsize(fn)
  mb_file_size = file_size / 1000000

  return mb_file_size

def load_data(nm, data_nm):
  '''
    Supported nm: An item in 
      exp_design['Name']
      group_design['Name']
      endo_design['Name']

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

  if level == 'endo':
    return load_endo_data(row, data_nm)

  scripts_base = [
    'g4_poswise_be', 
    'g5_combin_be', 
    'h4_poswise_del',
    'h6_anyindel',
  ]
  if data_nm in scripts_base:
    fn = prj_dir + '%s/out/%s/%s.pkl' % (row['Group'], data_nm, row['Local name'])
    with open(fn, 'rb') as f:
      data = pickle.load(f)

  if data_nm in ['e_newgenotype_Cas9']:
    fn = prj_dir + '%s/out/%s/%s.csv' % (row['Group'], data_nm, row['Local name'])
    data = pd.read_csv(fn, index_col = 0)

  scripts_adjust = [
    'ag4_poswise_be_adjust', 
    'ag5_combin_be_adjust', 
    'ae_newgenotype_Cas9_adjust',
    'ag4b_adjust_background',
    # 'ag4e_find_highfreq_batcheffects',
    'ag4a2_adjust_batch_effects',
    'ag4b4_filter_inprofile_batch_effects',
    'ag5a1a_hf_bc',
    'ag5a1b_subtract',
    'ag5a1c_ie',
    'ag5a2_adjust_batch_effects',
    'ag5a3_adjust_inprofile_batch_effects',
    'ag5a4_profile_subset',
    'ah6a1a_hf_bc',
    'ah6a1b_subtract',
    'ah6a3_remove_batch',
    'ah6c_reduce_1bp_indel_fq',
  ]
  if data_nm in scripts_adjust:
    if level == 'L1':
      fn = _config.OUT_PLACE + '%s/%s.pkl' % (data_nm, nm)
    elif level == 'L2':
      fn = _config.OUT_PLACE + 'b_group_L2/%s/%s.pkl' % (data_nm, nm)
    with open(fn, 'rb') as f:
      data = pickle.load(f)

  return data

def load_minq(nm, data_nm):
  '''
    Supported nm: An item in 
      exp_design['Name']
    
    (L1 only)

    Supported data_nm: [
      'g4_poswise_be', 
      'g5_combin_be', 
    ]
  '''

  if data_nm not in ['g4_poswise_be', 'g5_combin_be']:
    return None

  level, row = check_valid_and_fetch_level(nm)
  if level is None or level == 'L2':
    return None

  fn = prj_dir + '%s/out/%s/%s_minq.pkl' % (row['Group'], data_nm, row['Local name'])
  with open(fn, 'rb') as f:
    minq = pickle.load(f)

  return minq

def load_endo_data(row, data_nm):
  # Not designed to be called publicly, only from load_data.

  scripts_converter = {
    'g5_combin_be': 'g5_combin_be_single', 
    'h6_anyindel': 'h6_anyindel_single',
  }
  if data_nm in scripts_converter:
    data_nm_conv = scripts_converter[data_nm]
    fn = '/ahg/regevdata/projects/CRISPR-libraries/prj/lib-endo/%s/out/%s/%s.csv' % (row['Local dir'], data_nm_conv, row['Local name'])
    data = pd.read_csv(fn, index_col = 0)

  return data

##
# Dynamic condition matching
##
def get_library_endo_datasets(endo_nm_no_rep, data_nm):
  '''
    Returns a dict of dataframes.

    endo_nm_no_rep should be a member of 'Name (no rep)' in endo_design,
    and specifies a unique combination of cell-type, target site, and editor.

    Function 1) loads endogenous data from local directories, and 2) loads library data using _data.load_data() for all conditions matching the specified cell-type, target site, and editor. 

    Supports data_nm in [
      'g5_combin_be', 
      'h6_anyindel',
    ]

    Returns a dict of dataframes
  '''
  assert data_nm in ['g5_combin_be', 'h6_anyindel']

  all_data = dict()

  # Load endogenous datasets
  ed_subset = endo_design[endo_design['Name (no rep)'] == endo_nm_no_rep]
  for idx, row in ed_subset.iterrows():
    endo_nm = row['Name']
    data = load_data(endo_nm, data_nm)
    all_data[endo_nm] = data

  celltype = ed_subset['Local cell-type'].iloc[0]
  editor = ed_subset['Editor'].iloc[0]
  endo_loci_nm = ed_subset['Endogenous loci name'].iloc[0]

  '''
    Load library datasets.
    Search for all libraries that contain the endogenous loci. Match conditions with that library, cell-type, and editor.
  '''
  if endo_loci_nm not in set(endo_to_library['Endogenous loci name']):
    print(f'Failed to find {endo_loci_nm} in endo_to_library.csv')
    return all_data

  lib_row = endo_to_library[endo_to_library['Endogenous loci name'] == endo_loci_nm].iloc[0]
  libs = [col for col in endo_to_library.columns if col != 'Endogenous loci name']

  for lib_nm in libs:
    lib_seq_nm = lib_row[lib_nm]
    if type(lib_seq_nm) != str: continue

    for idx, row in treatment_control_design.iterrows():
      treat_nm = row['Treatment']
      match_lib_nm = bool(lib_nm == get_lib_nm(treat_nm))
      match_editor = bool(editor == get_editor_nm(treat_nm))

      if match_lib_nm and match_editor:
        print(f'Matched with {treat_nm}... ', end = '')
        d = load_data(treat_nm, data_nm)
        if lib_seq_nm not in d:
          print(f'but failed to find {lib_seq_nm}')
          continue
        single_target_d = d[lib_seq_nm]
        all_data[treat_nm] = single_target_d
        print(f'successfully loaded data')

  return all_data

##
# Data queries
##
def get_lib_design(nm):
  lib_nm = get_lib_nm(nm)
  lib_design = pd.read_csv(_config.DATA_DIR + 'library_%s.csv' % (lib_nm))
  seq_col = [s for s in lib_design.columns if 'Sequence context (' in s][0]
  return lib_design, seq_col

def get_g4_lib_design(nm):
  '''
    Returns a seq_col for each library design that guarantees that Protospacer position 0 index is identical for all target sites.
  '''
  lib_nm = get_lib_nm(nm)
  lib_design = pd.read_csv(_config.DATA_DIR + 'library_%s.csv' % (lib_nm))
  seq_col = [s for s in lib_design.columns if 'Sequence context (' in s][0]
  if lib_nm == 'CtoGA':
    seq_col = 'Sequence context prepended'
  return lib_design, seq_col

def get_public_lib_design():
  libs = ['12kChar', 'CtoT', 'AtoG', 'CtoGA']
  mdf = exp_design[exp_design['Library'].isin(libs)]

  ok_editors = [
    'ABE',
    'ABE-CP1040',
    'AID',
    'BE4',
    'BE4-CP1028',
    'BE4max_H47ES48A',
    'BE4_H47ES48A',
    'CDA',
    'eA3A',
    'evoAPOBEC',
    'eA3A_T31AT44A',
    'eA3A_T31A',
    'eA3Amax_T44DS45A',
  ]
  mdf = mdf[mdf['Editor'].isin(ok_editors)]

  ok_celltypes = ['mES', 'HEK293T', 'U2OS']
  mdf = mdf[mdf['Cell-type'].isin(ok_celltypes)]

  mdf['New name'] = mdf['Cell-type'] + '_' + mdf['Library'] + '_' + mdf['Editor'] + '_batch_' + mdf['Batch covariate (no celltype)']
  return mdf

def get_l1_nm(nm):
  level, row = check_valid_and_fetch_level(nm)
  if level is None:
    return None

  if level == 'L1':
    l1_nm = nm
  elif level == 'L2':
    row = group_design[group_design['Name'] == nm].iloc[0]
    l1_nm = row['Item1']
  return l1_nm

def get_new_name(nm):
  l1_nm = get_l1_nm(nm)

  row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
  return row['Name (new)']

def get_editor_type(nm):
  l1_nm = get_l1_nm(nm)

  row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
  editor_nm = row['Editor']

  editor_types = {
    'CtoTeditor': [
      'AID',
      'CDA',
      'BE4',
      'evoAPOBEC',
      'BE4-CP1028',
      'eA3A',
    ],
    'AtoGeditor': [
      'ABE',
      'ABE-CP1040',
    ],
    'Cas9': [
      'Cas9',
    ]
  }

  for typ in editor_types:
    if editor_nm in editor_types[typ]:
      return typ
  return None

def get_editor_nm(nm):
  l1_nm = get_l1_nm(nm)
  row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
  editor_nm = row['Editor']
  return editor_nm

def exp_design_nm_to_col(nm, col):
  l1_nm = get_l1_nm(nm)
  row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
  return row[col]

def get_lib_nm(nm):
  try:
    l1_nm = get_l1_nm(nm)
    row = exp_design[exp_design['Name'] == l1_nm].iloc[0]
    lib_nm = row['Library']
  except ValueError:
    brute_success = False
    for lib_nm_temp in zero_pos:
      if lib_nm_temp in nm:
        lib_nm = lib_nm_temp
        brute_success = True
    if not brute_success:
      raise ValueError
  return lib_nm

def get_ontarget_sites(lib_design, lib_nm):
  if lib_nm == '12kChar':
    otcats = ['satmut']
  elif lib_nm == 'AtoG':
    otcats = list(set(lib_design['Design category']))
  elif lib_nm == 'CtoT':
    otcats = list(set(lib_design['Design category']))
  elif lib_nm == 'LibA':
    otcats = list(set(lib_design['Design category']))
  elif lib_nm == 'CtoGA':
    otcats = list(set(lib_design['Design category']))

  return lib_design[lib_design['Design category'].isin(otcats)]['Name (unique)']

def get_disease_sites(lib_design, lib_nm):
  if lib_nm in ['CtoT', 'AtoG']:
    otcats = list(set(lib_design['Design category']))
  if lib_nm == 'CtoGA':
    otcats = ['Clinvar']

  return lib_design[lib_design['Design category'].isin(otcats)]['Name (unique)']


def get_replicates(celltype, lib_nm, editor_nm):
  crit = {
    'Cell-type': celltype, 
    'Library': lib_nm, 
    'Editor': editor_nm
  }
  match = lambda row: bool(sum([row[col_nm] == crit[col_nm] for col_nm in crit]) == len(crit))
  reps = exp_design[exp_design.apply(match, axis = 'columns')]['Name']
  return list(reps)


# Caching
lib_nms = dict()

'''
  Intention: These values should ONLY be used for g4 data structure. CtoGA library was designed with variable gRNA pos 0 index. To make it compatible with legacy code, I rewrote the g4_poswise_be data processing script to prepend a zero matrix to allow consistent gRNA pos 0 idx for that data structure, but the gRNA pos 0 idx values provided here do NOT work when indexing the designed target sequence. 
'''
zero_pos = {
  'LibA': 9,
  '12kChar': 21,
  'CtoT': 10,
  'AtoG': 10,
  'PAMvar': 12,
  'CtoGA': 30,
  # TODO: For short lib, it is 10. For long lib, it varies by target site.
}

def idx_to_pos(idx, nm):
  if nm not in lib_nms:
    lib_nms[nm] = get_lib_nm(nm)
  lib_nm = lib_nms[nm]
  if lib_nm == 'CtoGA':
    print('WARNING: Only use idx_to_pos on g4 data structure. Do not use it to index designed target sequence, particularly on CtoGA lib.')
  return idx - zero_pos[lib_nm]

def pos_to_idx(pos, nm):
  if nm not in lib_nms:
    try:
      lib_nms[nm] = get_lib_nm(nm)
    except ValueError:
      brute_success = False
      for lib_nm in zero_pos:
        if lib_nm in nm:
          lib_nms[nm] = lib_nm
          brute_success = True
      if not brute_success:
        raise ValueError
  lib_nm = lib_nms[nm]
  if lib_nm == 'CtoGA':
    print('WARNING: Only use idx_to_pos on g4 data structure. Do not use it to index designed target sequence, particularly on CtoGA lib.')
  return pos + zero_pos[lib_nm]

# Cache
pos_to_idx_safe_cache = dict()

def pos_to_idx_safe(pos, nm, target_nm):
  input_key = f'{pos}_{nm}_{target_nm}'
  if input_key in pos_to_idx_safe_cache:
    return pos_to_idx_safe_cache[input_key]

  if nm not in lib_nms:
    try:
      lib_nms[nm] = get_lib_nm(nm)
    except ValueError:
      brute_success = False
      for lib_nm in zero_pos:
        if lib_nm in nm:
          lib_nms[nm] = lib_nm
          brute_success = True
      if not brute_success:
        raise ValueError
  lib_nm = lib_nms[nm]
  lib_df = lib_dfs[lib_nm]
  pos0_idx = lib_df[lib_df['Name (unique)'] == target_nm]['Protospacer position zero index'].iloc[0]
  ans = pos + pos0_idx
  pos_to_idx_safe_cache[input_key] = ans
  return ans

