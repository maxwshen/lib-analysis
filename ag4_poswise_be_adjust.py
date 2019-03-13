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
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

##
# Functions
##
def adjust_treatment_control(treat_nm, control_nm):
  treat_data = _data.load_data(treat_nm, 'g4_poswise_be')
  control_data = _data.load_data(control_nm, 'g4_poswise_be')

  lib_design, seq_col = _data.get_lib_design(treat_nm)
  ''' 
    g4 format: data is a dict, keys = target site names
    values = np.array with shape = (target site len, 4)
      entries = int for num. Q30 observations
  '''

  adj_d = dict()
  stats_dd = defaultdict(list)
  
  nts = list('ACGT')
  mapper = {nts[s]: s for s in range(len(nts))}

  timer = util.Timer(total = len(lib_design))
  for idx, row in lib_design.iterrows():
    nm = row['Name (unique)']
    seq = row[seq_col]
    timer.update()

    stats_dd['Name'].append(nm)

    if nm not in treat_data:
      stats_dd['Status'].append('No treatment')
      continue

    t = treat_data[nm]
    if nm not in control_data:
      stats_dd['Status'].append('No control')
      adj_d[nm] = t
      continue

    stats_dd['Status'].append('Adjusted')
    c = control_data[nm]

    # Convert control to fractions
    c /= np.sum(c, axis = 1, keepdims = True)
    np.nan_to_num(c, 0)

    # Convert to readcounts using total treatment readcount as denominator  
    c *= np.sum(t, axis = 1, keepdims = True)

    # Zero out wild-type elements
    for jdx, ref_nt in enumerate(seq):
      c[jdx][mapper[ref_nt]] = 0

    # Adjust treatment by control
    t -= c

    # Set negative values to zero
    # Apply Q30 sequencing error filter.
    # Only trust mutations in treatment with max(3, 3*exp) reads than in control
    for jdx, ref_nt in enumerate(seq):
      tot = np.sum(t[jdx])
      exp_q30_errors = tot * 0.001 * 2
      threshold = max(3, exp_q30_errors)

      for kdx in range(len(t[jdx])):
        if kdx == mapper[ref_nt]:
          continue
        if t[jdx][kdx] < threshold:
          t[jdx][mapper[ref_nt]] += t[jdx][kdx]
          t[jdx][kdx] = 0
        if t[jdx][kdx] < 0:
          t[jdx][kdx] = 0

    adj_d[nm] = t

  with open(out_dir + '%s.pkl' % (treat_nm), 'wb') as f:
    pickle.dump(adj_d, f)

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + '%s_stats.csv' % (treat_nm))

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
    treat_nm, control_nm = row['Treatment'], row['Control']

    if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
      continue
    if 'Cas9' in treat_nm:
      continue

    command = 'python %s.py %s %s' % (NAME, treat_nm, control_nm)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, treat_nm, control_nm)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -l h_rt=4:00:00,h_vmem=1G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
def main(treat_nm = '', control_nm = ''):
  print(NAME)
  
  # Function calls
  if treat_nm == '' and control_nm == '':

    treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

    for idx, row in treat_control_df.iterrows():
      treat_nm, control_nm = row['Treatment'], row['Control']
      if 'Cas9' in treat_nm:
        continue
      print(treat_nm, control_nm)
      if os.path.exists(out_dir + '%s.pkl' % (treat_nm)):
        print('... already complete')
      else:
        adjust_treatment_control(treat_nm, control_nm)

  else:
    adjust_treatment_control(treat_nm, control_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(treat_nm = sys.argv[1], control_nm = sys.argv[2])
  else:
    gen_qsubs()
    # main()