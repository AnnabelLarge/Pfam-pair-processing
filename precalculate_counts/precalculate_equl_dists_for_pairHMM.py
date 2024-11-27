#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:37:34 2024

@author: annabel_large
"""
import jax
from jax import numpy as jnp
from functools import partial

import pandas as pd
import numpy as np
import os

from precalculate_counts.utils_for_precalc import (get_aa_counts)


def safe_convert_uint16(mat):
    """
    NOT jittable 
    """
    mat = mat.astype(int)
    try:
        assert (mat.min() >= 0) & (mat.max() <=65535)
        mat = mat.astype('uint16')
        return mat
    except:
        return mat

def get_eq_dist(alignment):
    """
    jit-compatible
    """
    AA_counts = get_aa_counts(alignment)
    
    non_gaps = jnp.where((alignment != 43) & (alignment != 0), 1, 0)
    match_pos = jnp.where(jnp.sum(non_gaps, axis=-1) == 2, True, False)[:,:,None]
    match_mask = jnp.broadcast_to(match_pos, 
                               ( match_pos.shape[0],
                                 match_pos.shape[1],
                                 2 )
                               )
    masked_alignment = alignment * match_mask
    del non_gaps, match_pos, match_mask
    
    AA_counts_subsOnly = get_aa_counts(masked_alignment)
    
    return AA_counts, AA_counts_subsOnly


@partial(jax.jit, static_argnums=(0,))
def precalc_equl_dist(batch_size, final_align):
    """
    jit-compatible
    """
    out = get_eq_dist(alignment = final_align)
    AA_counts, AA_counts_subsOnly = out
    del out
    
    assert AA_counts.shape == (20,)
    assert AA_counts_subsOnly.shape == (20,)
    
    out_dict = {'AAcounts': AA_counts,
                'AAcounts_subsOnly': AA_counts_subsOnly}
    return out_dict


def get_chunk_indices(num_total_samples, s):
    indices = []
    for i in range(0, num_total_samples, s):
        start = i
        end = min(i + s, num_total_samples)
        indices.append((start, end))
    return indices


def precalc_equl_dist_wrapper(splitname, batch_size):
    aligned_inputs_folder = f"{splitname}"
    
    pfam_files = [file for file in os.listdir(aligned_inputs_folder) if 
                  file.startswith('PF') and file.endswith('_aligned_mats.npy')]
    
    for file in tqdm(pfam_files):
        pfam_name = file.replace('_aligned_mats.npy','')
        
        with open(f'{pfam_folder}/{file}','rb') as f:
            aligned_inputs = np.load(f)
        
        # from this, only need first two inputs (aligned ancestor and descendant)
        align_with_bos_eos = aligned_inputs[:,:,[0,1]]
        
        
        ### also need to get rid of <bos> and <eos> (easier to do this in numpy)
        # remove <bos>
        align_np = align_with_bos_eos[:, 1:, :]
        
        # remove <eos> by replacing last token with 0
        eos_locs = np.where(align_np != 0, True, False).sum(axis=1)[:, 0] - 1
        align_np[ range(align_np.shape[0]), eos_locs, :] = 0
        
        
        ### convert to jax numpy array for jit-compiled functions
        align = jnp.array(align_np)
        
        # work in batches, in case output is too large
        idxes = get_chunk_indices(num_total_samples = align.shape[0], 
                                  s = batch_size)
        
        for i, (start, end) in enumerate(idxes):
            sub_mat = align[start:end, ...]
            out_dict = precalc_equl_dist(sub_mat.shape[0], sub_mat)
            
            # if you had to split into multiple batches, assign a part number
            # VERY IMPORTANT: when concatenating, concatenate IN NUMERICAL ORDER
            if len(idxes) > 1:
                prefix = pfam_name + f'-pt{i}'
            else:
                prefix = pfam_name
            
            for file_suffix, mat in out_dict.items():
                mat = safe_convert_uint16(mat)
                with open(f'{pfam_folder}/{prefix}_{file_suffix}.npy', 'wb') as g:
                    jnp.save(g, mat)


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import sys
    
    
    SPLITNAME = sys.argv[1]
    BATCH_SIZE = int( sys.argv[2] )
    
    ### run this on a machine with gpus
    precalc_equl_dist_wrapper(splitname = SPLITNAME, 
                              batch_size = BATCH_SIZE)
    