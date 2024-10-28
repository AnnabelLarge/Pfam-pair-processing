#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:01:40 2023

@author: annabel_large

sequences_paths is of shape: (B,L,[type]=3)
    > (dim2=0): ancestor
    > (dim2=1): descendant
    > (dim2=2): alignment-augmented descendant


Output in another 3D matrix: (B, L, [idxes]=2)
    > (dim3=0): m indices (position in the ANCESTOR)
    > (dim3=1): n indices (position in the DESCENDANT)

actions-
    > M: (m+1), (n+1)  [BOTH INCREASE]
    > I:  (m) , (n+1)  [INCREASE DESC]
    > D: (m+1),  (n)   [INCREASE ANC]

"""
import numpy as np

def safe_convert_int16(mat):
    assert mat.min() >= -32768
    assert mat.max() <= 32767
    return mat.astype('int16')


def generate_indices(full_mat, num_regular_toks=20, gap_tok = 43, align_pad = -9):
    # (B, L)
    mat = full_mat[:,1:,2]
    
    
    ### numpy where to find locations of matches, inserts, and deletes
    # padding: 0; need extra end padding to make dimensions work out
    pads = (mat == 0)
    extra_col = np.ones((pads.shape[0],1)).astype(bool)
    pads = np.concatenate([pads, extra_col], axis=1)
    
    
    # matches: 3 -> (num_regular_toks + 2)
    matches = ( ( mat >= 3 ) & 
                ( mat <= (num_regular_toks+2) ) 
              )
    
    # inserts: (num_regular_toks + 3) -> (2*num_regular_toks + 2) 
    inserts = ( ( mat >= (num_regular_toks+3) ) & 
                ( mat <= (2*num_regular_toks+2) ) 
              )
    
    # delete:  gap_tok
    deletes = (mat == gap_tok)
    
    
    ### some intermediate matrix to indicate what to add at (m,n) indices
    m_additions = np.zeros((mat.shape))
    n_additions = np.zeros((mat.shape))
    
    # matches: m+1, n+1
    m_additions[matches] += 1
    n_additions[matches] += 1
    
    # inserts: m+0, n+1
    n_additions[inserts] += 1
    
    # delete:  m+1, n+0
    m_additions[deletes] += 1
    
    # start alignment at (1,0)
    m_additions = np.concatenate([np.ones((m_additions.shape[0], 1)), 
                                  m_additions], axis=1)
    n_additions = np.concatenate([np.zeros((n_additions.shape[0], 1)), 
                                  n_additions], axis=1)
    
    
    ### loop to add m, n; this has to be done from start to end of alignment
    m_idx = np.cumsum(a=m_additions, axis=1)
    n_idx = np.cumsum(a=n_additions, axis=1)
    
    
    ### replace pad positions with placeholder value (-9)
    # this won't actually cause error in jax indexing, but do this for
    # my sanity
    m_idx[pads] = align_pad
    n_idx[pads] = align_pad
    
    
    ### concatenate
    indices = np.concatenate([m_idx[:,:,None],
                              n_idx[:,:,None]],
                             axis=2)
    
    indices_int16 = safe_convert_int16(indices)
    
    return indices_int16






if __name__ == '__main__':
    # import sys
    # from tqdm import tqdm
    
    # pre_lst = ['KPROT_OOD_VALID'] + [f'KPROT_split{i}' for i in range(5)]
    # # prefix = sys.argv[1]
    # #prefix = 'fiftySamps'
    
    # for prefix in tqdm(pre_lst):
    #     print(f'\n{prefix}')
        
    #     with open(f'{prefix}_sequences_paths.npy','rb') as f:
    #         in_mat = np.load(f)
        
    #     align_idxes = generate_indices(full_mat=in_mat, 
    #                                     num_regular_toks=20, 
    #                                     gap_tok = 43, 
    #                                     align_pad = -9)
        
    #     assert align_idxes.shape == (in_mat.shape[0],
    #                                   in_mat.shape[1],
    #                                   2)
        
        
    #     with open(f'{prefix}_align_idxes.npy','wb') as g:
    #         np.save(g, align_idxes)
    
    
    
    ################
    ### UNIT TESTS #
    ################
    def check_by_paper():
        """
        samp1:
        ======
        A - - C T G
        C T T C - G
        
        align idxes (m,n): 
            (1,0)
            (2,1)
            (2,2)
            (2,3)
            (3,4)
            (4,4)
            (5,5)
        
        
        samp2 (Exactly like Ian's protein example, 
        but with compressed alphabet):
        ===============================
        A - G T
        T C C T
        
        align idxes (m,n): 
            (1,0)
            (2,1)
            (2,2)
            (3,3)
            (4,4)
        
        
        samp3:
        ======
        - A -
        C A C
        
        align idxes (m,n):
            (1,0)
            (1,1)
            (2,2)
            (2,3)
            
            
        samp4:
        ======
        C A C
        - A -
        
        align idxes (m,n):
            (1,0)
            (2,0)
            (3,1)
            (4,1)
            
        """
        # (B=4, L=9, 3)
        align_aug1 = np.array([1,  5, 10, 10, 4, 43, 5, 2])
        align_aug2 = np.array([1,  6,  8,  4, 6,  2, 0, 0])
        align_aug3 = np.array([1,  8,  3,  8, 2,  0, 0, 0])
        align_aug4 = np.array([1, 43,  3, 43, 2,  0, 0, 0])
        
        align_aug = np.concatenate([align_aug1[None,:],
                                    align_aug2[None,:],
                                    align_aug3[None,:],
                                    align_aug4[None,:]], 
                                   axis=0)
        del align_aug1, align_aug2
        
        anc_seq = np.zeros(align_aug.shape)
        desc_seq = np.zeros(align_aug.shape)
        samp = np.concatenate([anc_seq[:,:,None],
                               desc_seq[:,:,None],
                               align_aug[:,:,None]], axis=-1)
        
        # (B=4, L=9, 2)
        true_out1 = np.array([[1, 2, 2, 2,  3,  4,  5, -9],
                              [0, 1, 2, 3,  4,  4,  5, -9]]).T
        
        true_out2 = np.array([[1, 2, 2, 3,  4, -9, -9, -9],
                              [0, 1, 2, 3,  4, -9, -9, -9]]).T
        
        true_out3 = np.array([[1, 1, 2, 2, -9, -9, -9, -9],
                              [0, 1, 2, 3, -9, -9, -9, -9]]).T
        
        true_out4 = np.array([[1, 2, 3, 4, -9, -9, -9, -9],
                              [0, 0, 1, 1, -9, -9, -9, -9]]).T
        
        true_out = np.concatenate([true_out1[None,:,:],
                                   true_out2[None,:,:],
                                   true_out3[None,:,:],
                                   true_out4[None,:,:]],
                                   axis=0)
        del true_out1, true_out2
        
        pred_out = generate_indices(full_mat=samp, 
                                    num_regular_toks=4, 
                                    gap_tok = 43, 
                                    align_pad = -9)
        
        assert np.allclose(pred_out, true_out)
        print('All cases worked! :)')
    
    
    
    
    def check_by_loop():
        """
        same examples as above
        """
        # (B=4, L=9, 3)
        align_aug1 = np.array([1,  5, 10, 10, 4, 43, 5, 2])
        align_aug2 = np.array([1,  6,  8,  4, 6,  2, 0, 0])
        align_aug3 = np.array([1,  8,  3,  8, 2,  0, 0, 0])
        align_aug4 = np.array([1, 43,  3, 43, 2,  0, 0, 0])
        
        align_aug = np.concatenate([align_aug1[None,:],
                                    align_aug2[None,:],
                                    align_aug3[None,:],
                                    align_aug4[None,:]], 
                                   axis=0)
        del align_aug1, align_aug2
        
        anc_seq = np.zeros(align_aug.shape)
        desc_seq = np.zeros(align_aug.shape)
        samp = np.concatenate([anc_seq[:,:,None],
                               desc_seq[:,:,None],
                               align_aug[:,:,None]], axis=-1)
        
        
        ### check this by hand
        true_out = []
        for b in range(samp.shape[0]):
            align_input = samp[b,:,2]
            
            m_idxes = [1]
            n_idxes = [0]
            
            for tok in align_input:
                # match
                if (tok >= 3) & (tok <= 6):
                    m_idxes.append(m_idxes[-1]+1)
                    n_idxes.append(n_idxes[-1]+1)
                
                # insert
                elif (tok >= 7) & (tok <= 11):
                    m_idxes.append(m_idxes[-1])
                    n_idxes.append(n_idxes[-1]+1)
                
                # delete
                elif (tok == 43):
                    m_idxes.append(m_idxes[-1]+1)
                    n_idxes.append(n_idxes[-1])
                
                # pad
                elif (tok == 0):
                    m_idxes.append(-9)
                    n_idxes.append(-9)
            
            # one last pad
            m_idxes.append(-9)
            n_idxes.append(-9)
            
            samp_out = [m_idxes, n_idxes]
            true_out.append(samp_out)
        
        
        true_out = np.array(true_out)
        true_out = true_out.transpose((0,2,1))
        
        
        pred_out = generate_indices(full_mat=samp, 
                                    num_regular_toks=4, 
                                    gap_tok = 43, 
                                    align_pad = -9)
        
        assert np.allclose(pred_out, true_out)
        print('calculation by simple loops is successful! :)')
    
    # # uncomment to do above unit tests
    # check_by_paper()
    # check_by_loop()
    
    
    