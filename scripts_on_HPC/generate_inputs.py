#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:38:41 2024

@author: annabel
"""
from Bio import Phylo
import pandas as pd
import numpy as np
from itertools import combinations
import random
import math
from copy import deepcopy

from scripts_on_HPC.precompute_align_idxes import generate_indices
from utils import make_sub_folder


def safe_int16(mat):
    if mat.dtype == 'int16':
        return mat
    assert mat.min() >= -32768
    assert mat.max() <= 32767
    return mat.astype('int16')

def safe_int8(mat):
    if mat.dtype == 'int8':
        return mat
    assert mat.min() >= -128
    assert mat.max() <= 127
    return mat.astype('int8')

def safe_int32(mat):
    if mat.dtype == 'int32':
        return mat
    assert mat.min() >= -2147483648
    assert mat.max() <= 2147483647
    return mat.astype('int32')
    
    
def read_inputs(pfam, seed_folder, trees_folder):
    raw_msa = {}
    pfam_level_meta = {'pfam': pfam,
                       'clan': '',
                       'type': ''}
    
    num_seqs = 0
    with open(f'{seed_folder}/{pfam}.seed','r') as f:
        for line in f:
            if line.startswith('#=GF CL'):
                pfam_level_meta['clan'] = line.strip().split()[-1]
            
            elif line.startswith('#=GF TP'):
                pfam_level_meta['type'] = line.strip().split()[-1]
            
            if not line.startswith('#'):
                num_seqs += 1
                name, seq = line.strip().split()
                seq = seq.upper()
                raw_msa[name] = seq
    
    pfam_level_meta['pfam_Nseqs'] = num_seqs
    
    tree = Phylo.read(f'{trees_folder}/{pfam}.tree', 'newick')
    return raw_msa, tree, pfam_level_meta

def dedup(tuple_list):
    return list( set( tuple( sorted(t) ) for t in tuple_list ) )

def read_pairs_from_file(filename):
    df = pd.read_csv(filename, sep='\t')
    
    # added this bit to specifically handle my inputs
    df = df[['seq1','seq2']]
    
    pairs = df.itertuples(index=False, name=None)
    pairs = dedup(pairs)
    return pairs

def generate_random_pairs(seqnames, percent_of_pairs, filename_of_cherries):
    cherries = read_pairs_from_file(filename_of_cherries)
    reverse_cherries = []
    for (seq1, seq2) in cherries:
        reverse_cherries.append( (seq2, seq1) )
    
    banned_tuples = set( cherries + reverse_cherries )
    
    all_possible_pairs = [tup for tup in combinations(seqnames, 2) if 
                          tup not in banned_tuples]
    
    num_to_sample = math.ceil(percent_of_pairs * len(all_possible_pairs))
    pairs = random.sample(all_possible_pairs, num_to_sample)
    return pairs
    
def extract_alignment(ancestor, descendant, raw_msa):
    anc_gapped = raw_msa[ancestor]
    desc_gapped = raw_msa[descendant]
    alignment = []
    num_matches = 0
    num_subs = 0
    num_ins = 0
    num_dels = 0
    for tup in zip(anc_gapped, desc_gapped):
        if tup != ('.','.'):
            alignment.append(tup)
            
            # ins
            if tup[0] == '.' and tup[1] != '.':
                num_ins += 1
            
            # del
            elif tup[0] != '.' and tup[1] == '.':
                num_dels += 1
            
            # exact match
            elif tup[0] == tup[1]:
                num_matches += 1
            
            # subs
            else:
                num_subs += 1
    
    anc_seq_len = len( anc_gapped.replace('.','') )
    desc_seq_len = len( desc_gapped.replace('.','') )
    alignment_len = len( alignment )
    psi = num_matches/min(anc_seq_len, desc_seq_len)
    
    out_dict = {'perc_seq_id': psi,
                'anc_seq_len': anc_seq_len,
                'desc_seq_len': desc_seq_len,
                'alignment_len': alignment_len,
                'num_matches': num_matches,
                'num_subs': num_subs,
                'num_ins': num_ins,
                'num_dels': num_dels}
    return alignment, out_dict

def get_alphabet():
    special = ['<pad>', '<bos>', '<eos>']
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mapping = {elem:i for i,elem in enumerate(special + aas)}
    mapping['.'] = 43
    return mapping

def reversible_featurizer(str_alignment, mapping, max_len):
    # first sample is forward, second sample is reverse
    raw_neural_format = np.zeros( (2, max_len, 3), dtype='int8' )
    hmm_align_format = np.zeros( (2, max_len, 2), dtype='int8' )
    
    def update_buckets( b, align_idx, anc_char, desc_char, anc_pos, desc_pos):
        # deletion
        if (desc_char == '.') & (anc_char != '.'):
            raw_neural_format[b, anc_pos, 0] = mapping[anc_char]
            anc_pos += 1
            
            raw_neural_format[b, align_idx, 2] = mapping['.']
            
            hmm_align_format[b, align_idx, 0] = mapping[anc_char]
            hmm_align_format[b, align_idx, 1] = mapping['.']
            
        # insertion 
        elif (anc_char == '.') & (desc_char != '.'):
            raw_neural_format[b, desc_pos, 1] = mapping[desc_char]
            desc_pos += 1
            
            raw_neural_format[b, align_idx, 2] = mapping[desc_char] + 20
            
            hmm_align_format[b, align_idx, 0] = mapping['.']
            hmm_align_format[b, align_idx, 1] = mapping[desc_char]
            
        # match 
        elif (anc_char != '.') & (desc_char != '.'):
            raw_neural_format[b, anc_pos, 0] = mapping[anc_char]
            anc_pos += 1
            raw_neural_format[b, desc_pos, 1] = mapping[desc_char]
            desc_pos += 1
            
            raw_neural_format[b, align_idx, 2] = mapping[desc_char]
            
            hmm_align_format[b, align_idx, 0] = mapping[anc_char]
            hmm_align_format[b, align_idx, 1] = mapping[desc_char]
        
        return anc_pos, desc_pos
    
    
    assert len(str_alignment) < PAD_TO

    fw_anc_pos = 0
    fw_desc_pos = 0
    rv_anc_pos = 0
    rv_desc_pos = 0
    for align_idx, (seq1_char, seq2_char) in enumerate(str_alignment):
        # forward: (seq1, seq2)
        fw_anc_pos, fw_desc_pos = update_buckets(b = 0,
                                                 align_idx = align_idx,
                                                 anc_char = seq1_char, 
                                                 desc_char = seq2_char, 
                                                 anc_pos = fw_anc_pos, 
                                                 desc_pos = fw_desc_pos)
        
        # reverse: (seq2, seq1)
        rv_anc_pos, rv_desc_pos = update_buckets(b = 1,
                                                 align_idx = align_idx,
                                                 anc_char = seq2_char, 
                                                 desc_char = seq1_char, 
                                                 anc_pos = rv_anc_pos, 
                                                 desc_pos = rv_desc_pos)
        
        
    return raw_neural_format, hmm_align_format

def encode_one_pair(i, seq1, seq2, tree, raw_msa, pfam):
    dist = tree.distance(seq1,seq2)
    fw_pair_level_metadata = {'pairID': f'FW_{pfam}_p{i}',
                              'ancestor': seq1,
                              'descendant': seq2,
                              'TREEDIST_anc-to-desc': dist}
    
    rv_pair_level_metadata = {'pairID': f'RV_{pfam}_p{i}',
                              'ancestor': seq2,
                              'descendant': seq1,
                              'TREEDIST_anc-to-desc': dist}
    
    str_alignment, add_to_fw = extract_alignment(ancestor = seq1, 
                                                 descendant = seq2, 
                                                 raw_msa = raw_msa)
    fw_pair_level_metadata = {**fw_pair_level_metadata, **add_to_fw}
    
    # swap info between anc and desc for reverse pair
    add_to_rv = {'perc_seq_id': add_to_fw['perc_seq_id'],
                 'anc_seq_len': add_to_fw['desc_seq_len'],
                 'desc_seq_len': add_to_fw['anc_seq_len'],
                 'alignment_len': add_to_fw['alignment_len'],
                 'num_matches': add_to_fw['num_matches'],
                 'num_subs': add_to_fw['num_subs'],
                 'num_ins': add_to_fw['num_dels'],
                 'num_dels': add_to_fw['num_ins']
                 }
    rv_pair_level_metadata = {**rv_pair_level_metadata, **add_to_rv}
    del add_to_fw, add_to_rv
    
    
    # generate neural and hmm pair alignment inputs in one go
    mapping = get_alphabet()
    raw_neural_format, hmm_align_format = reversible_featurizer(str_alignment = str_alignment, 
                                                                mapping = mapping, 
                                                                max_len = PAD_TO)
    
    return (fw_pair_level_metadata, 
            rv_pair_level_metadata, 
            raw_neural_format, 
            hmm_align_format)


def validate_eos(raw_neural_format, seqlens):
    # <eos> should only appear once
    assert np.allclose(np.where(raw_neural_format == 2, 1, 0).sum(axis=1),
                       np.ones((raw_neural_format.shape[0], raw_neural_format.shape[2]), dtype='int8')
                       )
    
    # seqlens should be the same as before
    new_seqlens = np.where((raw_neural_format != 0) & (raw_neural_format != 2), 
                           1, 
                           0).sum(axis=1)
    assert np.allclose(new_seqlens, seqlens)
    
    # padding tokens should appear AFTER <eos>, not before
    lengths = np.arange(raw_neural_format.shape[1], dtype='int16')[None, :, None]
    lengths = np.broadcast_to(lengths, raw_neural_format.shape)
    
    padding_idxes = np.where(raw_neural_format==0, 
                             lengths,
                             9999)
    start_of_padding = padding_idxes.min(axis=1)
    assert np.allclose(start_of_padding - seqlens,
                       np.ones(start_of_padding.shape))

def add_bos_eos(raw_neural_format, validate=True):
    new_eos_col = np.zeros( (raw_neural_format.shape[0], 
                             1, 
                             raw_neural_format.shape[2]), dtype='int8' )
    with_eos = np.concatenate([raw_neural_format, new_eos_col], axis=1)
    seqlens = np.where(with_eos != 0, 1, 0).sum(axis=1)
    
    with_eos[range(with_eos.shape[0]), seqlens[:,0], 0] = 2
    with_eos[range(with_eos.shape[0]), seqlens[:,1], 1] = 2
    with_eos[range(with_eos.shape[0]), seqlens[:,2], 2] = 2
    
    if validate:
        validate_eos(with_eos, seqlens)
    
    new_bos_col = np.ones( (with_eos.shape[0], 
                             1, 
                             with_eos.shape[2]), dtype='int8' )
    
    neural_format = np.concatenate([new_bos_col, with_eos], axis=1)
    
    return neural_format


def featurize_one_pfam(pfam, seed_folder, trees_folder, pairs_from, filename,
                       percent_of_pairs = None):
    ### read inputs, get pairs
    raw_msa, tree, pfam_level_metadata = read_inputs(pfam = pfam, 
                                                     seed_folder = seed_folder, 
                                                     trees_folder = trees_folder)
    
    if pairs_from == 'rand_samp':
        pairs = generate_random_pairs(seqnames = raw_msa.keys(), 
                                      percent_of_pairs = percent_of_pairs,
                                      filename_of_cherries = filename)
        
    elif pairs_from == 'file':
        pairs = read_pairs_from_file(filename = filename)
    
    
    ### if you don't find any, exit function
    if len(pairs) == 0:
        return None
    
    
    ### iterate through pairs
    metadata = []
    raw_neural_format = []
    hmm_align_format = []
    for pair_id, (seq1, seq2) in enumerate(pairs):
        out = encode_one_pair(i = pair_id, 
                              seq1 = seq1, 
                              seq2 = seq2, 
                              tree = tree,
                              raw_msa = raw_msa,
                              pfam = pfam)
        metadata.append(out[0])
        metadata.append(out[1])
        raw_neural_format.append(out[2])
        hmm_align_format.append(out[3])
    
    metadata = pd.DataFrame(metadata)
    raw_neural_format = np.concatenate(raw_neural_format, axis=0)
    hmm_align_format = np.concatenate(hmm_align_format, axis=0)
    
    
    ### add pfam level info to metadata
    for key, val in pfam_level_metadata.items():
        metadata[key] = val
    
    
    ### update and add to neural inputs
    # add <eos>, <bos> to neural inputs
    neural_format = add_bos_eos(raw_neural_format = raw_neural_format, 
                                validate = False)
    
    
    del raw_msa, tree, pfam_level_metadata, pairs, pair_id, seq1, seq2, out
    del raw_neural_format
    
    # precompute alignment indices for neural inputs
    neural_align_idxes = generate_indices(full_mat = neural_format, 
                                          num_regular_toks=20, 
                                          gap_tok = 43, 
                                          align_pad = -9)
    
    # ### create different pairHMM input
    # align_len = (hmm_align_format != 0).sum(axis=1)[:,0]
    # out = summarize_alignment(hmm_align_format, align_len, gap_tok=43)
    # subCounts_persamp = out[0]
    # insCounts_persamp = out[1]
    # delCounts_persamp = out[2]
    # transCounts_persamp = out[3]
    # del out, align_len
    
    # AA_counts = get_aa_counts(hmm_align_format)
    
    # non_gaps = np.where((hmm_align_format != 43) & (hmm_align_format != 0), 1, 0)
    # match_pos = np.where(np.sum(non_gaps, axis=-1) == 2, True, False)[:,:,None]
    # match_mask = np.repeat(match_pos, 2, axis=-1)
    # masked_hmm_align_format = hmm_align_format * match_mask
    # del non_gaps, match_pos, match_mask
    
    # AA_counts_subsOnly = get_aa_counts(masked_hmm_align_format)
    
    
    neural_dset = {'sequences_paths': safe_int8(neural_format),
                    'align_idxes': safe_int16(neural_align_idxes)}
    
    hmm_pair = {'pair_alignments': safe_int8(hmm_align_format)}
    
    # hmm_precalc = {'subsCounts': subCounts_persamp,
    #                'insCounts': insCounts_persamp,
    #                'delCounts': delCounts_persamp,
    #                'transCounts': transCounts_persamp,
    #                'AAcounts': AA_counts,
    #                'AAcounts_subsOnly': AA_counts_subsOnly}
    
    return neural_dset, hmm_pair, metadata 



##################################
### gather random pairs; combine #
##################################
def make_rand_samp(pfam, 
                   seed_folder,
                   trees_folder,
                   percent_of_pairs,
                   file_of_cherries,
                   dset_prefix):
    out = featurize_one_pfam(pfam = pfam, 
                             seed_folder = seed_folder,
                             trees_folder = trees_folder, 
                             pairs_from = 'rand_samp',
                             percent_of_pairs = percent_of_pairs,
                             filename = file_of_cherries)
    
    if out != None:
        neural_dset = out[0]
        hmm_pair = out[1]
        metadata = out[2]
        
        ### neural
        folder = f'{dset_prefix}_neural'
        for file_suffix, mat in neural_dset.items():
            with open(f'{folder}/{pfam}_{file_suffix}.npy','wb') as g:
                np.save(g, mat)
        del folder
        
        
        ### hmm pair align
        folder = f'{dset_prefix}_hmm_pairAlignments'
        for file_suffix, mat in hmm_pair.items():
            with open(f'{folder}/{pfam}_{file_suffix}.npy', 'wb') as g:
                np.save(g, mat)
        del folder
        
        
        ### metadata (copy to every folder)
        folder = f'{dset_prefix}_all_metadata'
        metadata.to_csv(f'{folder}/{pfam}_metadata.tsv', sep='\t')



##########################################################
### generate pairs from an input file (usually cherries) #
##########################################################
def samples_from_file(pfam, 
                      seed_folder,
                      trees_folder,
                      filename,
                      dset_prefix):
    out = featurize_one_pfam(pfam = pfam, 
                             seed_folder = seed_folder,
                             trees_folder = trees_folder, 
                             pairs_from = 'file',
                             filename = filename)
    
    if out != None:
        neural_dset = out[0]
        hmm_pair = out[1]
        metadata = out[2]
        
        ### neural
        folder = f'{dset_prefix}_neural'
        for file_suffix, mat in neural_dset.items():
            with open(f'{folder}/{pfam}_{file_suffix}.npy','wb') as g:
                np.save(g, mat)
        del folder
        
        
        ### hmm pair align
        folder = f'{dset_prefix}_hmm_pairAlignments'
        for file_suffix, mat in hmm_pair.items():
            with open(f'{folder}/{pfam}_{file_suffix}.npy', 'wb') as g:
                np.save(g, mat)
        del folder
        
        
        ### metadata (copy to every folder)
        folder = f'{dset_prefix}_all_metadata'
        metadata.to_csv(f'{folder}/{pfam}_metadata.tsv', sep='\t')





###############################################################################
### Ran on savio CPU cluster in parallel, so I called this   ################## 
### script directly                                          ##################
### TODO: make these general function wrappers               ##################
###############################################################################
RANDOM_SEED = 2
random.seed(RANDOM_SEED)
PAD_TO = 4100

if __name__ == '__main__':
    import os
    import sys
    
    # RANDOM_SEED = 2
    # random.seed(RANDOM_SEED)
    # PAD_TO = 4100
    
    
    pfam_filename = sys.argv[1]
    # pfam_filename = 'pfams_in_OOD_valid.tsv'
    
    splitname = pfam_filename.replace('.tsv','').split('_')[-1]
    
    rand_dset_prefix = f'FIVEPERC-RAND_{splitname}'
    cherries_dset_prefix = f'CHERRIES_{splitname}'
    percent_of_pairs = 0.05
    
    
    with open(f'pfams_in_parts/{pfam_filename}','r') as f:
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
    
    
    
