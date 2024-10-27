#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 02:32:12 2024

@author: annabel
"""
import os
from remove_duplicates import (find_repeats,
                               parse_within_families_file,
                               parse_across_families_file,
                               remove_samples)

from clean_short_peps_invalid_chars import clean_short_peps_invalid_chars
from prune_trees_after_msa_clean import prune_trees_after_msa_clean

from utils import find_missing_info


def clean_pfam_data()
find_repeats('seed_alignments')
dict1 = parse_within_families_file()
dict2 = parse_across_families_file()
remove_samples('seed_alignments', dict1)
remove_samples('seed_alignments', dict2)
clean_short_peps_invalid_chars('seed_alignments')

prune_trees_after_msa_clean('trees')

missing_trees, missing_msas = find_missing_info(seed_alignment_dir = 'seed_alignments',
                                                tree_dir = 'trees')

err_msg = f'Have tree files without matching seed alignments?\n{missing_msas}'
assert len(missing_msas) == 0, err_msg

if len(missing_trees) > 0:
    with open(f'ALIGN_IN_FASTTREE.tsv', 'w') as g:
        [g.write(elem + '\n') for elem in missing_trees]

