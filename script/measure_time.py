'''
    Yuxuan, 27 June
    Script for measuring time cost.
'''

import os
import sys

# describe which cases to run and how many times
run_dict = {
    9 : [500, 1000, 2000],
    10 : [500, 1000, 2000],
    11 : [500, 1000, 2000],
    12 : [500, 1000, 2000]
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
