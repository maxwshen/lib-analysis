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
endo_reverse_CtoGA_design = pd.read_csv(_config.DATA_DIR + 'endo_reverse_CtoGA_design.csv', index_col = 0)

##
#
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

##
#
##
def is_not_wildtype(row):
  nt_cols = [col for col in row.index if col != 'Count']
  num_edits = sum([bool(row[col] != col[0]) for col in nt_cols])
  return bool(num_edits != 0)


def get_edited_nts(row, edit_row):
  nt_cols = [col for col in edit_row.index if col != 'Count']
  seq = row['Amplicon (NGG orientation)']
  pos1_idx = row['gRNA_pos1_idx (NGG orientation)']
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

  snp_idx = design_row['Position of SNP in gRNA']
  snp_col = f'C{snp_idx}'
  corrected_nt = design_row['Correction edit'][-1]

  denom = sum(df[df[snp_col] != 'C']['Count'])
  stats['Obs. gt induce precision with any bystander precision'] = sum(df[df[snp_col] == corrected_nt]['Count']) / denom if denom > 0 else np.nan

  induce_precision_count = 0
  nt_cols = [col for col in df.columns if col != 'Count']
  for idx, edit_row in df.iterrows():
    num_edits = sum([bool(nt_col[0] == edit_row[nt_col] for nt_col in nt_cols)])
    if num_edits == 1 and edit_row[snp_col] == corrected_nt:
      induce_precision_count += edit_row['Count']

  denom = sum(df['Count'])
  stats['Obs. gt induce precision'] = induce_precision_count / denom if denom > 0 else np.nan

  aa_stats = {
    'Edited AA': 0,
    'Wild-type AA': 0,
    'Goal AA': 0,
  }
  for idx, edit_row in df.iterrows():
    obs_30nt = get_edited_nts(design_row, edit_row)

    obs_aas = nts_to_aas(obs_30nt, design_row['AA frame position'], snp_idx, design_row['AA strand relative to gRNA'])

    if obs_aas == design_row['AA sequence - reference']:
      pass
    else:
      aa_stats['Edited AA'] += edit_row['Count']

    if obs_aas == design_row['AA sequence - pathogenic']:
      aa_stats['Goal AA'] += edit_row['Count']

  denom = sum(df['Count'])
  stats['Obs. aa induce precision'] = aa_stats['Goal AA'] / denom if denom > 0 else np.nan

  denom = aa_stats['Edited AA']
  stats['Obs. aa induce precision among edited aas'] = aa_stats['Goal AA'] / denom if denom > 0 else np.nan

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
    Uses endo_pathogenic_lib.csv and endo_pathogenic_design.csv, linking local name conditions to endogenous loci in "lib".

    Input: g5 from endo_fulledits_stats.
  '''
  print(NAME)
  
  # Function calls
  dd = defaultdict(list)
  timer = util.Timer(total = len(endo_reverse_CtoGA_design))
  for idx, row in endo_reverse_CtoGA_design.iterrows():
    endo_loci_nm = row['Endogenous loci name']
    dfs = endo_design[endo_design['Endogenous loci name'] == endo_loci_nm]
    for nm in dfs['Name']:
      stats = calc_stats(nm, row)
      if stats is None:
        continue
      for col in stats:
        dd[col].append(stats[col])
      for col in row.index:
        dd[col].append(row[col])
    timer.update()

  df = pd.DataFrame(dd)
  df.to_csv(out_dir + 'endo_pathogenic_all_stats.csv')

  return

if __name__ == '__main__':
  if len(sys.argv) > 1:
    main()
  else:
    # gen_qsubs()
    main()