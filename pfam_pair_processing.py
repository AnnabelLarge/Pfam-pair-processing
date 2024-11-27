#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:17:33 2024

@author: annabel
"""
import argparse
import sys

from initial_cleaning.initial_cleaning import main as initial_cleaning_fn
from prepare_for_featurization.make_splits import make_splits
from prepare_for_featurization.prepare_for_featurization import main as split_n_pick 
from concatenate_parts.concatenate_parts import main as concat_parts

parser = argparse.ArgumentParser(
                    prog='Pfam-pair-processing',
                    description='Run different parts of the data cleaning pipeline')

valid_tasks = ['initial_cleaning', 
               'split_data',
               'pick_cherries_split_data',
               'concat_parts']

parser.add_argument('-task',
                    type = str,
                    required = True,
                    choices = valid_tasks,
                    help = '(str) Choose a task')


### arguments used in multiple places
parser.add_argument('-pfam_seed_file',
                    type = str,
                    help = '(str) Name of the original single seed file')


### arguments required for inital_cleaning
parser.add_argument('-header',
                    type = str,
                    help = '(str) Header to add to output stats file')


### arguments used in split_data and pick_cherries_split_data
parser.add_argument('-num_splits',
                    type = int,
                    default = 10,
                    help = '(int) number of splits (not including OOD valid)')

parser.add_argument('-rand_key',
                    type = int,
                    default = 6,
                    help = '(int) random key for randomly selecting data splits')

parser.add_argument('-topk1_valid',
                    type = int,
                    default = 3,
                    help = '(int) number of widest pfams for OOD valid')

parser.add_argument('-topk2_valid',
                    type = int,
                    default = 8,
                    help = '(int) number of gappiest pfams for OOD valid')


### extra argument for pick_cherries_split_data
parser.add_argument('-tree_dir',
                    type = str,
                    help = '(str) the folder of .tree files from PFam+FastTree')


### arguments for concatenate_parts
parser.add_argument('-splitname',
                    type=str,
                    help ='(str) the name of the split to process')

parser.add_argument('-alphabet_size',
                    type=int,
                    default=20,
                    help ='(int) base alphabet size; 20 for amino acids')

args = parser.parse_args()



if args.task == 'initial_cleaning':
    initial_cleaning_fn(pfam_seed_file = args.pfam_seed_file,
                        header = args.header)

elif args.task == 'split_data':
    make_splits(pfam_seed_file = args.pfam_seed_file,
               num_splits = args.num_splits,
               rand_key = args.rand_key,
               topk1_valid = args.topk1_valid,
               topk2_valid = args.topk2_valid)

elif args.task == 'pick_cherries_split_data':
    # combination of picke trees and make_splits
    split_n_pick(pfam_seed_file = args.pfam_seed_file,
                 tree_dir = args.tree_dir,
                 num_splits = args.num_splits,
                 rand_key = args.rand_key,
                 topk1_valid = args.topk1_valid,
                 topk2_valid = args.topk2_valid)

elif args.task == 'concat_parts':
    concat_parts(splitname = args.splitname,
                 alphabet_size = args.alphabet_size)

