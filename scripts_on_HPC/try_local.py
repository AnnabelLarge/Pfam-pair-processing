#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:25:05 2024

@author: annabel
"""
import os
import sys
import random
from scripts_on_HPC.generate_inputs import (make_rand_samp,
                                            samples_from_file)
from utils import make_sub_folder

RANDOM_SEED = 2
random.seed(RANDOM_SEED)
PAD_TO = 4100


split_file_lst = ['pfams_in_OOD_valid.tsv',
                  'pfams_in_split0.tsv',
                  'pfams_in_split1.tsv']

for pfam_filename in split_file_lst:
    splitname = pfam_filename.replace('.tsv','').split('_')[-1]
    
    rand_dset_prefix = f'TEST-RAND_{splitname}'
    cherries_dset_prefix = f'TEST-CHERRIES_{splitname}'
    percent_of_pairs = 0.05
    
    with open(f'{pfam_filename}','r') as f:
        pfams_in_split = [line.strip() for line in f]
    
    file_lst = [f'CHERRIES-FROM_trees/{pf}_cherries.tsv' 
                for pf in pfams_in_split]
    
    for suffix in ['neural', 'hmm_pairAlignments', 'all_metadata']: #, 'precalculated_counts']:
        rand_folder = f'{rand_dset_prefix}_{suffix}'
        cherries_folder = f'{cherries_dset_prefix}_{suffix}'
        make_sub_folder(in_dir = '.', sub_folder=rand_folder)
        make_sub_folder(in_dir = '.', sub_folder=cherries_folder)
    
    for i in range(len(pfams_in_split)):
        if i % 10 == 0:
            print(f'{i}/{len(pfams_in_split)}')
            
        pfam = pfams_in_split[i]
        file_of_cherries = file_lst[i]
        
        
        # skip this one for now
        if pfam == 'PF10417':
            continue
        
        
        ### random sample (NOT including cherries)
        make_rand_samp(pfam = pfam, 
                       seed_folder = 'seed_alignments',
                       trees_folder = 'trees',
                       file_of_cherries = file_of_cherries,
                       percent_of_pairs = percent_of_pairs,
                       dset_prefix = rand_dset_prefix)
        
        
        ### all possible cherries
        samples_from_file(pfam = pfam, 
                          seed_folder = 'seed_alignments',
                          trees_folder = 'trees',
                          filename = file_of_cherries,
                          dset_prefix = cherries_dset_prefix)
        