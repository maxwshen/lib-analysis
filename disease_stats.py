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
inp_dir = _config.OUT_PLACE + 'ag5a4_profile_subset/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# df
##
def subset_edited_rows(df, nt_cols):
  edited_idxs = []
  col_to_ref_nt = {col: col[0] for col in nt_cols}
  for idx, row in df.iterrows():
    # Treat . as wild-type. Require row to have a high-confidence edit
    num_edits = sum([bool(row[col] != col_to_ref_nt[col]) & bool(row[col] != '.') for col in nt_cols])
    if num_edits > 0:
      edited_idxs.append(idx)
  return df.loc[edited_idxs]

def impute_dot_as_wildtype(df, nt_cols):
  for col in nt_cols:
    ref_nt = col[0]
    df.loc[df[col] == '.', col] = ref_nt
  return df

def remove_noisy_edits(df, nt_cols, exp_nm):
  '''
    Remove rows with high frequency among edited rows that only have mutations far from base edit region 
  '''
  abe_flag = bool('ABE' in exp_nm) or bool('AtoG' in exp_nm)
  if abe_flag:
    core_nt = 'A'
  else:
    core_nt = 'C'

  core_cols = ['%s%s' % (core_nt, pos) for pos in range(1, 10 + 1)]

  df['Frequency'] = df['Count'] / np.sum(df['Count'])
  ok_idxs = []
  ts_core_cols = set(nt_cols) & set(core_cols)
  col_to_ref_nt = {col: col[0] for col in nt_cols}
  for idx, row in df.iterrows():
    num_core_edits = sum([bool(row[col] != col_to_ref_nt[col]) for col in ts_core_cols])
    if num_core_edits == 0 and row['Frequency'] >= 0.025:
      pass
    else:
      ok_idxs.append(idx)

  return df.loc[ok_idxs]

##
# Statistics
##
def get_precise_gt_correction_count(df, nt_cols, snp_pos, correct_nt, path_nt):
  corrected_count = 0
  for idx, row in df.iterrows():
    num_edits = sum([bool(row[nt_col] != nt_col[0]) for nt_col in nt_cols])
    has_correct_edit = bool(row[f'{path_nt}{snp_pos}'] == correct_nt) 
    if num_edits == 1 and has_correct_edit:
      corrected_count += row['Count']
  return corrected_count

##
# Amino acids
##
triplet_to_aa = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W', 
} 

def nts_to_aas(seq_30nt, aa_frame, path_pos_wrt_grna, aa_strand_relative_to_seq):
  if aa_frame == 'intronic':
    return ''

  path_idx = path_pos_wrt_grna + 9

  if aa_strand_relative_to_seq == '-':
    seq_30nt = compbio.reverse_complement(seq_30nt)
    path_idx = len(seq_30nt) - path_idx - 1

  # aa_frame in [1, 2, 3] -> [0, 1, 2]
  aa_frame_0idx = int(aa_frame) - 1
  begin_frame = (aa_frame_0idx - path_idx) % 3
  first_triplet_start_idx = (3 - begin_frame) % 3

  aas = ''
  for idx in range(first_triplet_start_idx, len(seq_30nt) + 3, 3):
    triplet = seq_30nt[idx : idx + 3]
    if len(triplet) != 3:
      break
    aa = triplet_to_aa[triplet]
    # print(triplet, aa, idx)
    aas += aa

  return aas

def edit_row_to_seq_30nt(design_row, edit_row, seq_col):
  nt_cols = [col for col in edit_row.index if col != 'Count' and col != 'Frequency']
  seq = design_row[seq_col]
  pos1_idx = design_row['Protospacer position zero index'] + 1
  seq_30nt = seq[pos1_idx - 10 : pos1_idx + 20]

  pos0_idx = 9
  for col in nt_cols:
    ref_nt = col[0]
    obs_nt = edit_row[col]
    pos = int(col[1:])
    seq_idx = pos + pos0_idx
    assert seq_30nt[seq_idx] == ref_nt
    seq_30nt = seq_30nt[:seq_idx] + obs_nt + seq_30nt[seq_idx + 1:]

  return seq_30nt

##
# Form groups
##
def form_data(exp_nm, start_idx, end_idx):
  data = _data.load_data(exp_nm, 'ag5a4_profile_subset')
  lib_design, seq_col = _data.get_lib_design(exp_nm)
  lib_nm = _data.get_lib_nm(exp_nm)
  disease_nms = _data.get_disease_sites(lib_design, lib_nm)

  # Subset for dumb parallelization, ensure only disease target sites used
  lib_design = lib_design.iloc[start_idx : end_idx + 1]
  lib_design = lib_design[lib_design['Name (unique)'].isin(disease_nms)]

  nms = lib_design['Name (unique)']
  seqs = lib_design[seq_col]
  nm_to_seq = {nm: seq for nm, seq in zip(nms, seqs)}

  stats_dd = defaultdict(list)

  nms_shared = [nm for nm in nms if nm in data]
  timer = util.Timer(total = len(nms_shared))
  for iter, nm in enumerate(nms_shared):

    df = data[nm]
    seq = nm_to_seq[nm]

    design_row = lib_design[lib_design['Name (unique)'] == nm].iloc[0]
    snp_pos = int(design_row['Position of SNP in gRNA'])
    correct_nt = design_row['Corrected nucleotide (gRNA orientation)']
    path_nt = design_row['Pathogenic nucleotide (gRNA orientation)']

    nt_cols = [col for col in df.columns if col != 'Count' and col != 'Frequency']

    # Impute . as wildtype
    df = impute_dot_as_wildtype(df, nt_cols)
    total_ct = sum(df['Count'])

    # Ensure each row is unique
    df = df.groupby(nt_cols)['Count'].agg('sum').reset_index()

    # Filter unedited columns
    df = subset_edited_rows(df, nt_cols)
    edited_ct = sum(df['Count'])

    df = remove_noisy_edits(df, nt_cols, exp_nm)

    gt_correct_ct = get_precise_gt_correction_count(df, nt_cols, snp_pos, correct_nt, path_nt)

    ## Overall statistics
    stats_dd['Name (unique)'].append(nm)

    stats_dd['Obs. correction count'].append(gt_correct_ct)
    stats_dd['Obs. total count'].append(total_ct)
    stats_dd['Obs. edited count'].append(edited_ct)

    stats_dd['Obs. gt correct fraction in all reads'].append(gt_correct_ct / total_ct if total_ct > 0 else np.nan)
    stats_dd['Obs. gt correct precision in edited reads'].append(gt_correct_ct / edited_ct if edited_ct > 0 else np.nan)
    stats_dd['Obs. editing frequency'].append(edited_ct / total_ct if total_ct > 0 else np.nan)

    # Amino acid correction for CtoGA
    if 'AA sequence - reference' in design_row.index and type(design_row['AA sequence - reference']) == str:

      orients = list('-+')
      d1 = bool(design_row['Designed orientation w.r.t. genome'] == '+')
      d2 = bool(design_row['AA frame strand'] == '+')
      xor_int = int(d1 == d2)
      aa_strand_relative_to_seq = orients[xor_int]

      aa_stats = {
        'Unedited AA': 0,
        'Edited AA': 0,
        'Goal AA': 0,
      }
      if design_row['AA sequence - pathogenic'] != design_row['AA sequence - reference']:
        for jdx, edit_row in df.iterrows():
          seq_30nt = edit_row_to_seq_30nt(design_row, edit_row, seq_col)
          obs_aas = nts_to_aas(seq_30nt, design_row['AA frame position'], snp_pos, aa_strand_relative_to_seq)

          pp0idx = design_row['Protospacer position zero index']
          seq_30nt_path = design_row[seq_col][pp0idx - 9 : pp0idx + 21]
          aa_path_with_bc = nts_to_aas(seq_30nt_path, design_row['AA frame position'], snp_pos, aa_strand_relative_to_seq)

          seq_30nt_wt = seq_30nt_path[:9 + snp_pos] + design_row['Corrected nucleotide (gRNA orientation)'] + seq_30nt_path[9 + snp_pos + 1:]
          aa_wt_with_bc = nts_to_aas(seq_30nt_wt, design_row['AA frame position'], snp_pos, aa_strand_relative_to_seq)

          if obs_aas == aa_path_with_bc:
            aa_stats['Unedited AA'] += edit_row['Count']
          else:
            aa_stats['Edited AA'] += edit_row['Count']

          if obs_aas == aa_wt_with_bc:
            aa_stats['Goal AA'] += edit_row['Count']


      stats_dd['Obs. aa correct precision among edited gts'].append(
        aa_stats['Goal AA'] / edited_ct if edited_ct > 0 else np.nan
      )
      stats_dd['Obs. aa correct precision among edited aas'].append(
        aa_stats['Goal AA'] / aa_stats['Edited AA'] if aa_stats['Edited AA'] > 0 else np.nan
      )
      stats_dd['Obs. aa correct precision among all reads'].append(
        aa_stats['Goal AA'] / total_ct if total_ct > 0 else np.nan
      )
      if stats_dd['Obs. aa correct precision among edited gts'] < stats_dd['Obs. gt correct precision in edited reads']:
        import code; code.interact(local=dict(globals(), **locals()))

    else:
      stats_dd['Obs. aa correct precision among edited gts'].append(np.nan)
      stats_dd['Obs. aa correct precision among edited aas'].append(np.nan)
      stats_dd['Obs. aa correct precision among all reads'].append(np.nan)

    timer.update()

  # Save
  stats_df_collected = pd.DataFrame(stats_dd)

  stats_df = lib_design.merge(
    stats_df_collected, 
    on = 'Name (unique)', 
    how = 'outer',
  )

  stats_df.to_csv(out_dir + '%s_%s_%s_stats.csv' % (exp_nm, start_idx, end_idx))
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
    if lib_nm not in ['CtoT', 'AtoG', 'CtoGA']:
      continue

    if lib_nm != 'CtoGA':
      continue

    if lib_nm in ['CtoT', 'AtoG']:
      end_idx = 12000
      jump = 2000
    elif lib_nm == 'CtoGA':
      end_idx = 4000
      jump = 500

    for start_idx in range(0, end_idx, jump):
      end_idx = start_idx + jump - 1

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
def main(exp_nm = '', start_idx = '', end_idx = ''):
  print(NAME)
  
  # Function calls
  form_data(exp_nm, int(start_idx), int(end_idx))

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1], start_idx = sys.argv[2], end_idx = sys.argv[3])
  else:
    gen_qsubs()
    # main()