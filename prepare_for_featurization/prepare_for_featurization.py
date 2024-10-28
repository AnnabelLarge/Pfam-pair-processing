#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:12:26 2024

@author: annabel
"""
import os

from prepare_for_featurization.cherry_picking_fns import pick_trees_serially
from prepare_for_featurization.make_splits import make_splits


def prepare_for_featurization(pfam_seed_file: str,
                              tree_dir: str,
                              num_splits: int = 10,
                              rand_key: int = 6,
                              topk1_valid: int = 3,
                              topk2_valid: int = 8):
    """
    wrapper to find cherries and make PFam/Clan-level splits
    
    option for embarrassingly parallelizable workflow:
    --------------------------------------------------
     1. split tree_dir into multiple parts
     2. use prepare_for_featurization.cherry_picking_fns.greedy_cherry_picker() 
        on each part of tree_dir, in parallel
     3. bring parts back together
     4. use make_splits() at end
    """
    
    pick_trees_serially(tree_dir = tree_dir,
                        draw_trees = False)
    
    make_splits(pfam_seed_file = pfam_seed_file,
                num_splits = num_splits,
                rand_key = rand_key,
                topk1_valid = topk1_valid,
                topk2_valid = topk2_valid)


### testing
if __name__ == '__main__':
    prepare_for_featurization(pfam_seed_file = 'EXAMPLE_Pfam-A.seed',
                              tree_dir = 'trees',
                              num_splits = 2,
                              topk1_valid = 1,
                              topk2_valid = 0)

