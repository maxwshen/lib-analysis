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
inp_dir = _config.OUT_PLACE
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)
exp_design_df = pd.read_csv(_config.DATA_DIR + 'master_exp_design.csv', index_col = 0)

max_1nt_indel_fq = 0.30

##
# Primary logic
##
def reduce_1bp_indel_fq(exp_nm):
  '''
  '''
  data = None
  if exp_nm in set(treat_control_df['Treatment']):
    data = _data.load_data(exp_nm, 'ah6a1b_subtract')

  if data is None:
    print('Error : could not load data')
    import code; code.interact(local=dict(globals(), **locals()))
    sys.exit(1)

  lib_design, seq_col = _data.get_lib_design(exp_nm)

  dd = defaultdict(list)
  timer = util.Timer(total = 56)
  for indel_len in range(-40, 15+1):
    ah6b_fn = _config.OUT_PLACE + f'ah6b_calc_indels_global/{exp_nm}_pos_melt_{indel_len}nt.csv'
    df = pd.read_csv(ah6b_fn, index_col = 0)
    
    if 'Frequency' in df.columns:
      count_col = 'Frequency'
    else:
      count_col = 'Count'
    dd['Count'].append(np.sum(df[count_col]))
    dd['Indel length'].append(indel_len)
    timer.update()

  df = pd.DataFrame(dd)
  df['Frequency'] = df['Count'] / np.sum(df['Count'])

  fq_1nt_indel = sum(df[df['Indel length'].isin([-1, 1])]['Frequency'])
  print(exp_nm, fq_1nt_indel)

  reduce_factor = 1
  if fq_1nt_indel > max_1nt_indel_fq:
    '''
      Assert that after reduction, global fraction of 1-nt indels is no greater than max_1nt_indel_fq.

      final_fq = max_1nt_indel_fq
      af = adjusted frequency of 1nt indels ()
      r = remainder (tot. frequency of non 1-bp indels)
      
      final_fq = af / (af + r)
      final_fq(af + r) = af
      final_fq + r*finalfq/af = 1
      r*finalfq/af = 1 - final_fq
      af = r*final_fq / (1 - final_fq)
    '''
    adj_frac = (1 - fq_1nt_indel) * max_1nt_indel_fq / (1 - max_1nt_indel_fq)
    reduce_factor = adj_frac / fq_1nt_indel
    print(reduce_factor)

    def reduce_count(row):
      if row['Indel length'] != 1:
        return row['Count']
      else:
        return row['Count'] * reduce_factor

    timer = util.Timer(total = len(data))
    for target_nm in data:
      df = data[target_nm]
      if 1 in df['Indel length']:
        df['Count'] = df.apply(reduce_count, axis = 'columns')
        data[target_nm] = df
      timer.update()

  with open(out_dir + f'{exp_nm}.pkl', 'wb') as f:
    pickle.dump(data, f)

  stats_df = pd.DataFrame({
    'Reduce factor': reduce_factor,
    'Condition': exp_nm,
    }, index = [0],
  )
  stats_df.to_csv(out_dir + f'stats_{exp_nm}.csv')

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

  treatments = list(set(treat_control_df['Treatment']))
  all_conds = treatments

  num_scripts = 0
  for cond in all_conds:

    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, cond)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -V -P regevlab -l h_rt=12:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

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
  
  reduce_1bp_indel_fq(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()