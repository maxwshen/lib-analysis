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
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

# Default params
inp_dir = _config.OUT_PLACE + 'be_combin_general_annot/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

master_exp_design = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)
treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Primary
##
def gather_statistics(celltype, lib_nm, editor_nm):
  print(celltype, lib_nm, editor_nm)
  [rep1, rep2] = _data.get_replicates(celltype, lib_nm, editor_nm)

  df1 = pd.read_csv(inp_dir + '%s.csv' % (rep1), index_col = 0)
  df2 = pd.read_csv(inp_dir + '%s.csv' % (rep2), index_col = 0)

  lib_nm = _data.get_lib_nm(rep1)
  lib_design, seq_col = _data.get_lib_design(rep1)
  ontarget_sites = _data.get_ontarget_sites(lib_design, lib_nm)

  # Prepare data
  # data = data[data['Total count'] >= 100]
  df1 = df1[df1['Name (unique)'].isin(ontarget_sites)]
  df2 = df2[df2['Name (unique)'].isin(ontarget_sites)]

  id_cols = [
    'Name (unique)',
    'gRNA (20nt)',
    seq_col,
  ]
  mdf = df1.merge(df2, on = id_cols, suffixes = ['_r1', '_r2'])

  stat_col = 'Fraction edited'
  mdf['absdiff'] = np.abs(mdf['%s_r1' % (stat_col)] - mdf['%s_r2' % (stat_col)])

  mdf['abslfc'] = np.abs(np.log2(mdf['%s_r1' % (stat_col)]) - np.log2(mdf['%s_r2' % (stat_col)]))


  n_col = 'Total count'
  mdf['Total n'] = mdf['%s_r1' % (n_col)] + mdf['%s_r2' % (n_col)]

  mdf.to_csv(out_dir + '%s_%s_%s.csv' % (celltype, lib_nm, editor_nm))
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
  num_scripts = 0
  
  editors = set(master_exp_design['Editor'])
  editors.remove('Cas9')
  editors.remove('UT')
  celltypes = ['HEK293T', 'mES', 'U2OS']
  libraries = ['12kChar', 'CtoT', 'AtoG']

  for celltype in celltypes:
    for lib_nm in libraries:
      for editor_nm in editors:

        reps = _data.get_replicates(celltype, lib_nm, editor_nm)

        if len(reps) != 2:
          continue

        command = 'python %s.py %s %s %s' % (NAME, celltype, lib_nm, editor_nm)
        script_id = NAME.split('_')[0]

        # Write shell scripts
        sh_fn = qsubs_dir + 'q_%s_%s_%s_%s.sh' % (script_id, celltype, lib_nm, editor_nm)
        with open(sh_fn, 'w') as f:
          f.write('#!/bin/bash\n%s\n' % (command))
        num_scripts += 1

        # Write qsub commands
        qsub_commands.append('qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
def main(args):
  print(NAME)
  
  if len(args) != 0:
    # Run single
    [celltype, lib_nm, editor_nm] = args
    gather_statistics(celltype, lib_nm, editor_nm)

  else:
    # Run all

    editors = set(master_exp_design['Editor'])
    editors.remove('Cas9')
    editors.remove('UT')
    celltypes = ['HEK293T', 'mES', 'U2OS']
    libraries = ['12kChar', 'CtoT', 'AtoG']

    for celltype in celltypes:
      for lib_nm in libraries:
        for editor_nm in editors:

          reps = _data.get_replicates(celltype, lib_nm, editor_nm)

          if len(reps) != 2:
            continue

          gather_statistics(celltype, lib_nm, editor_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    # gen_qsubs()
    main([])