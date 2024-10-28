#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:09:58 2024

@author: annabel
"""
from concatenate_parts.concatenate_parts import concatenate_parts

for setname in ['TEST-CHERRIES','TEST-RAND']:
    for split in ['valid', 'split0','split1']:
        splitname = f'{setname}_{split}'
        include_pair_align = False
        
        concatenate_parts(splitname = splitname,
                           alphabet_size = 20,
                           include_pair_align = include_pair_align)