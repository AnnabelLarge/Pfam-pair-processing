#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:10:59 2024

@author: annabel
"""
import os
import pandas as pd
import numpy as np


def sort_pfams(folder_name: str):
    all_files = [f for f in os.listdir(folder_name) if
                 f.startswith('PF')]
    
    pfams_sorted = [f.split('_')[0].split('-')[0] for f in all_files]
    pfams_sorted.sort()
    
    return pfams_sorted


def sort_pfam_parts(folder_name: str,
                    pfam: str,
                    file_suffix: str):
    pfam_parts_in_folder = [f for f in os.listdir(folder_name) 
                            if f.startswith(pfam) and f.endswith(file_suffix)]
    num_parts = len(pfam_parts_in_folder)
    
    if num_parts > 1:
        sorted_pfams_prefixes = [f'{pfam}-pt{i}' for i in range(num_parts)]
    
    elif num_parts == 1:
        sorted_pfams_prefixes = [pfam]
    
    return sorted_pfams_prefixes


def combine_metadata(splitname: str):
    """
    concatenate the metadata for all training pairs; this is used in both
      EvolPairHMMore and DogShow
    
    usually the metadata dataframes for each PFam aren't split into parts...
      so don't worry about that yet?
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
    
    returns:
    --------
        - pfams_in_order (list): the sorted list of pfams; use this
                                 to establish the concatenation order for all
                                 other files
    
    outputs:
    --------
        - concatenated metadata about the training set
    """
    folder = f'{splitname}_all_metadata'
    
    out_meta = []
    pfams_in_order = sort_pfams(folder_name = folder)
    for pfam_prefix in pfams_in_order:
        sub_df = pd.read_csv(f'{folder}/{pfam_prefix}_metadata.tsv',
                             sep='\t',
                             index_col=0)
        sub_df['original_loc'] = f'{folder}/{pfam_prefix}_metadata.tsv'
        out_meta.append(sub_df)
    
    out_meta = pd.concat(out_meta)
    out_meta = out_meta.reset_index(drop=True)
    out_meta.to_csv(f'{splitname}_metadata.tsv', sep='\t')
    
    return pfams_in_order


def combine_neural_inputs(splitname: str,
                          pfams_in_order: list):
    """
    concatenate the inputs for DogShow
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined precursors for DogShow
    """
    folder = f'{splitname}_neural'
    
    out_seqs_paths = []
    out_align_idxes = []
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_sequences_paths.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            with open(f'{folder}/{pfam_prefix}_sequences_paths.npy','rb') as f:
                out_seqs_paths.append( np.load(f) )
            
            with open(f'{folder}/{pfam_prefix}_align_idxes.npy','rb') as f:
                out_align_idxes.append( np.load(f) )
    
    out_seqs_paths = np.concatenate(out_seqs_paths, axis=0)
    out_align_idxes = np.concatenate(out_align_idxes, axis=0)
    
    with open(f'{splitname}_sequences_paths.npy','wb') as g:
        np.save(g, out_seqs_paths)
    
    with open(f'{splitname}_align_idxes.npy','wb') as g:
        np.save(g, out_align_idxes)
    

def combine_hmm_align_inputs(splitname: str,
                             pfams_in_order: list):
    """
    concatenate the precursors for EvolPairHMM (technically could 
        put these into EvolPairHMM too? But I haven't tried that in
        a while, so... better safe than sorry)
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined precursors for EvolPairHMM
    """
    folder = f'{splitname}_hmm_pairAlignments'
    
    out_mat = []
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_pair_alignments.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            with open(f'{folder}/{pfam_prefix}_pair_alignments.npy','rb') as f:
                out_mat.append( np.load(f) )
        
    out_mat = np.concatenate(out_mat, axis=0)
    
    with open(f'{splitname}_pair_alignments.npy','wb') as g:
        np.save(g, out_mat)
        
        
def combine_hmm_precalc_inputs(splitname: str,
                               pfams_in_order: list,
                               alphabet_size: int = 20):
    """
    concatenate/sum the inputs for EvolPairHMM
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - pfams_in_order (list): the list of file prefixes in order, 
                                 usually pfam+part_id
        - alphabet_size (int=20): the base alphabet size; 20 for proteins
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined inputs for EvolPairHMM
    """
    folder = f'{splitname}_hmm_precalc_counts'
    
    all_out = {'subCounts': [],
               'insCounts': [],
               'delCounts': [],
               'transCounts': [],
               'AAcounts': np.zeros( (alphabet_size,) ),
               'AAcounts_subsOnly': np.zeros( (alphabet_size,) )
               }
    
    for pfam in pfams_in_order:
        sorted_pfam_in_parts = sort_pfam_parts(folder_name = folder,
                                               pfam = pfam,
                                               file_suffix = '_transCounts.npy')
        
        for pfam_prefix in sorted_pfam_in_parts:
            # these get concatenated
            with open(f'{folder}/{pfam_prefix}_subCounts.npy','rb') as f:
                all_out['subCounts'].append( np.load(f) )
            
            with open(f'{folder}/{pfam_prefix}_insCounts.npy','rb') as f:
                all_out['insCounts'].append( np.load(f) )
            
            with open(f'{folder}/{pfam_prefix}_delCounts.npy','rb') as f:
                all_out['delCounts'].append( np.load(f) )
                
            with open(f'{folder}/{pfam_prefix}_transCounts.npy','rb') as f:
                all_out['transCounts'].append( np.load(f) )
            
            # these get added
            with open(f'{folder}/{pfam_prefix}_AAcounts.npy','rb') as f:
                all_out['AAcounts'] = all_out['AAcounts'] + np.load(f)
            
            with open(f'{folder}/{pfam_prefix}_AAcounts_subsOnly.npy','rb') as f:
                all_out['AAcounts_subsOnly'] = (all_out['AAcounts_subsOnly'] + 
                                                np.load(f) )
        
    for suffix in ['subCounts','insCounts','delCounts','transCounts']:
        to_write = np.concatenate(all_out[suffix])
        with open(f'{splitname}_{suffix}.npy','wb') as g:
            np.save(g, to_write)
    
    for suffix in ['AAcounts','AAcounts_subsOnly']:
        with open(f'{splitname}_{suffix}.npy','wb') as g:
            np.save(g, to_write)
        


def main(splitname: str,
         alphabet_size: int = 20,
         include_pair_align: bool = False):
    """
    run the functions above to combine file parts
    
    input:
    ------
        - splitname (str): the folder prefix, usually setname+foldname
        - alphabet_size (int=20): the base alphabet size; 20 for proteins
        - include_pair_align (bool=False): whether or not to concatenate this 
                                           folder of intermediates (usually no)
    
    returns:
    --------
        (None)
    
    outputs:
    --------
        - concatenated/combined inputs for EvolPairHMM and DogShow
    """
    
    # concatenate metadata, and use this to get the file ordering
    pfams_in_order = combine_metadata(splitname = splitname)
    
    # concatenate everything else
    combine_neural_inputs(splitname = splitname,
                          pfams_in_order = pfams_in_order)
    combine_hmm_precalc_inputs(splitname = splitname,
                               pfams_in_order = pfams_in_order,
                               alphabet_size = alphabet_size)
    
    # usually don't use raw hmm pair alignments as inputs for anything,
    #   but if you want to concatenate it, go ahead
    if include_pair_align:
        combine_hmm_align_inputs(splitname = splitname,
                                 pfams_in_order = pfams_in_order)
