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
inp_dir = _config.OUT_PLACE + 'fulledits_general_correct1bpfq/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)


##
# Form groups
##
def get_statistics(exp_nm):
  '''
    Complete global editing statistics for each target site:
      - Base editing frequency among all non-noisy reads
      - Indel frequency among all non-noisy reads
    Can be used to calculate base editing to indel ratio (low biological replicability, due to low replicability of indel frequency).
  '''

  stats_df = pd.read_csv(inp_dir + f'{exp_nm}.csv', index_col = 0)

  mdf = stats_df

  # Filter readcount
  mdfs = mdf[mdf['Total count_indel'] >= 1000]

  # Filter target sites with no substrate nt in core
  has_c_in_core = lambda row: bool('C' in row['gRNA (20nt)'][3 : 7])
  mdfs['Has Core C'] = mdfs.apply(has_c_in_core, axis = 'columns')
  mdfs = mdfs[mdfs['Has Core C']]

  # Filter target sites with very low base editing fq
  mdfs['Base edit fq'] = (mdfs['Edited count']) / mdfs['Total count_indel']
  mdfs = mdfs[mdfs['Base edit fq'] > 0.025]
  # mdfs = mdfs[mdfs['Base edit fq'] > 0.01]

  # No pseudocounts
  mdfs = mdfs[mdfs['Indel count'] > 0]
  # # Pseudocount
  # mdfs['Edited count'] += 1
  # mdfs['Indel count'] += 1

  mdfs['Base edit to indel ratio'] = mdfs['Edited count'] / mdfs['Indel count']

  mdfs['Log10 base edit to indel ratio'] = np.log10(mdfs['Base edit to indel ratio'])
  mdfs.to_csv(out_dir + '%s.csv' % (exp_nm))

  from scipy.stats.mstats import gmean
  data = mdfs['Base edit to indel ratio']
  stats = {
    'Num. target sites': len(mdfs),
    'Geometric mean': gmean(data),
    'Median': np.median(data),
    '25th percentile': np.percentile(data, 25),
    '75th percentile': np.percentile(data, 75),
    'Geometric std': np.exp(np.std(np.log(data))),
  }

  return stats

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
    if 'Cas9' in cond:
      continue

    command = 'python %s.py %s' % (NAME, cond)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, cond)
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
def main(exp_nm = ''):
  print(NAME)
  
  stats = get_statistics(exp_nm)

  stats_df = pd.DataFrame(stats, index = [0])
  stats_df.to_csv(out_dir + f'{exp_nm}_bootstrap.csv')

  # # Function calls
  # be_treatments = [nm for nm in treat_control_df['Treatment'] if 'Cas9' not in nm]
  # timer = util.Timer(total = len(be_treatments))
  # stats_dd = defaultdict(list)
  # for treat_nm in be_treatments:
  #   stats = get_statistics(treat_nm)
  #   for col in stats:
  #     stats_dd[col].append(stats[col])
  #   stats_dd['Condition'].append(treat_nm)
  #   timer.update()

  # stats_dd.to_csv(out_dir + f'base_edit_to_indel_ratio_stats.csv')

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()