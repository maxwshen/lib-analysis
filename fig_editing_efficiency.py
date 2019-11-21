# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
from scipy.stats import binom

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
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # command = f'cp {inp_dir}* {out_dir}'
  # ans = subprocess.check_output(command, shell = True)
  # print('Copied')

  dd = defaultdict(list)

  be_treatments = [nm for nm in treat_control_df['Treatment'] if 'Cas9' not in nm]
  timer = util.Timer(total = len(be_treatments))
  for treat_nm in be_treatments:
    df = pd.read_csv(inp_dir + f'{treat_nm}.csv', index_col = 0)

    ee = df['Base edit freq'].dropna()

    dd['Condition'].append(treat_nm)
    dd['Editing efficiency - mean'].append(np.mean(ee))
    dd['Editing efficiency - std'].append(np.std(ee))
    dd['Editing efficiency - median'].append(np.median(ee))
    dd['Editing efficiency - 25th percentile'].append(np.percentile(ee, 25))
    dd['Editing efficiency - 75th percentile'].append(np.percentile(ee, 75))
    dd['Num. target sites with >100 reads'].append(sum(df['Total count_indel'] >= 100))
    dd['Total reads'].append(np.sum(df['Total count_indel']))

    timer.update()

  stats_df = pd.DataFrame(dd)
  stats_df.to_csv(out_dir + 'editing_efficiency.csv')

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(treat_nm = sys.argv[1])
  # else:
  #   gen_qsubs()
  main()