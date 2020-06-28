'''
    Yuxuan, 27 June
    Script for measuring time cost.
'''

import os
import sys

# describe case ids and runs times
run_dict = {
    # small system
    7 : [512, 1024, 2048],
    8 : [512, 1024, 2048],

    # large system
    9 : [64, 128, 256, 512],
    10 : [64, 128, 256],
    11 : [32, 64],
    12 : [32]
}

os.chdir('../bin')

for case_id, run_times_list in run_dict.items():
    for run_times in run_times_list:
        print(f'Case {case_id}: ', end='')
        print(os.popen(
            f'./serial ../data/case{case_id}.txt ../sresult/res{case_id}.txt {run_times}').read(), end='')

        print(f'Case {case_id}: ', end='')
        print(os.popen(
            f'./parallel ../data/case{case_id}.txt ../presult/res{case_id}.txt {run_times}').read(), end='')
