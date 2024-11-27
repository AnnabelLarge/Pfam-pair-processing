#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:19:15 2024

@author: annabel
"""
import jax
from jax import numpy as jnp

from generate_inputs.generate_inputs import (reversible_featurizer,
                                             get_alphabet)



"""
TEST OUT THE NEW REVERSIBLE FEATURIZER


sample 1:
=========
forward alignment:
    - A A
    C C -

reverse alignment:
    C C -
    - A A

forward indices:
    (1,0)
    (1,1)
    (2,2)
    (3,2)

reverse indices:
    (1,0)
    (2,0)
    (3,1)
    (3,2)
    

matrix formats:
================
unaligned_seqs_matrix = (B, L_seq, 2)
  - dim2=0: ancestor, unaligned
  - dim2=1: descendant, unaligned
  - L_seq INCLUDES <bos>, <eos>

aligned_seqs_matrix = (B, L_align, 4)
  - dim2=0: ancestor GAPPED (aligned)
  - dim2=1: descendant GAPPED (aligned)
  - dim2=2: precomputed m indexes for neural models
  - dim2=3: precomputed n indexes for neural models

"""
### sample 1 input
str_alignment = [('.','C'), 
                 ('A','C'),
                 ('A','.')]


### forward outputs
fw_true_unaligned_out = jnp.array([[1,3,3,2,0],
                                   [1,4,4,2,0]]).T[None,:,:]

fw_true_aligned_out = jnp.array([[1, 43, 3,  3,  2],
                                 [1,  4, 4, 43,  2],
                                 [1,  1, 2,  3, -9],
                                 [0,  1, 2,  2, -9]]).T[None,:,:]


### reverse outputs
rv_true_unaligned_out = jnp.array([[1,4,4,2,0],
                                   [1,3,3,2,0]]).T[None,:,:]

rv_true_aligned_out = jnp.array([[1,  4, 4, 43,  2],
                                 [1, 43, 3,  3,  2],
                                 [1,  2, 3,  3, -9],
                                 [0,  0, 1,  2, -9]]).T[None,:,:]

true_unaligned_out = jnp.concatenate([fw_true_unaligned_out,
                                      rv_true_unaligned_out],
                                     axis=0)


### concatenate
true_unaligned_out = jnp.concatenate([fw_true_unaligned_out,
                                      rv_true_unaligned_out],
                                     axis=0)
del fw_true_unaligned_out, rv_true_unaligned_out

true_aligned_out = jnp.concatenate([fw_true_aligned_out,
                                    rv_true_aligned_out],
                                    axis=0)
del fw_true_aligned_out, rv_true_aligned_out


### test
out = reversible_featurizer(str_alignment = str_alignment, 
                            mapping = get_alphabet(), 
                            max_len = 3)

pred_unaligned_out, pred_aligned_out = out
del out

assert jnp.allclose(pred_aligned_out, true_aligned_out)
assert jnp.allclose(pred_unaligned_out, true_unaligned_out)
