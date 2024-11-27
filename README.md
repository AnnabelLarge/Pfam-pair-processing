# Curating pair alignment training data from PFam seed alignments
## Requirements
### Database, Software Versions
- FTP from [Pfam v36.0](https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/)  
- Use FastTree version 2.1.11 No SSE3 to impute missing trees 

### Python packages
Using **Python 3.9.18** and standard packages that come with a miniconda installation. Additional non-standard packages you may have to install include:  

- **Jax 0.4.28:** for precomputing transition and emission counts (for pairHMM inputs).
- **Biopython 1.81:** for processing phlogenetic trees  

### Helpful computational resources
Access to an HPC cluster is helpful for processing PFam inputs in parallel. I used the Savio cluster at UC Berkeley.  

Code to precompute transition and emission counts (for pairHMM inputs) is implemented in Jax and can optionally be done on a GPU.


## Recreate Training Data
#### 1.) Download seed alignments file
```
wget "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/Pfam-A.seed.gz" && \
  gunzip Pfam-A.seed.gz
```

#### 2.) Download trees (which were generated with FastTree)
```
wget "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/trees.tgz" && \
  tar -xzvf trees.tgz && \
  mv nfs/production/agb/pfam/sequpd/36.0/Release/36.0/trees .
```

#### 3.) Run initial cleaning on seed alignments file
```
python pfam_pair_processing.py \
  -task initial_cleaning \
  -pfam_seed_file Pfam-A.seed \
  -header "# Alignments from PFam v36.0" 
```

#### 4.) If any pfams do not come with trees, use FastTree with default options to impute trees. Use standard FastTree workflow for this.

#### 5.) Generate list of cherries from each pfam, and make data splits by either-
    
*5a.)* Generating list of cherries with `prepare_for_featurization.cherry_picking_fns.greedy_cherry_picker` on multiple tree directories in parallel, combining results, then calling-
```
python pfam_pair_processing.py \
  -task split_data \
  -pfam_seed_file Pfam-A.seed \
  -num_splits [NUM_SPLITS] \
  -rand_key [RAND_KEY] \
  -topk1_valid [TOPK1_VALID] \
  -topk2_valid [TOPK2_VALID]
```
**OR**  
  
*5b.)* Picking cherries serially on one tree directory, then using split_data
```
python pfam_pair_processing.py \
  -task pick_cherries_split_data \
  -pfam_seed_file Pfam-A.seed \
  -tree_dir [TREE_DIR] \
  -num_splits [NUM_SPLITS] \
  -rand_key [RAND_KEY] \
  -topk1_valid [TOPK1_VALID] \
  -topk2_valid [TOPK2_VALID]
```
#### 6.) Move `generate_inputs.py` to current working directory. Alter and run- 

```
python generate_inputs.py [PFAMS_IN_SPLIT_TEXT-FILE]
```

 for all splits. `[PFAMS_IN_SPLIT_TEXT-FILE]` will be a text file that contains the names of all pfams in the given split.  

**Highly** recommended to do this in parallel.

#### 7.) Either precalculate all transition/emission counts and equlibrium distributions (7a), OR only equilibrium distributions (7b). Outputs from 7a are needed in order to use the `-have_precalculated_counts` flag in EvolPairHMM. Otherwise, outputs (7b) are sufficient.

*7a.)* Move `precalculate_counts_for_pairHMM.py` to current working directory, and run-
```
python precalculate_counts_for_pairHMM.py [SPLIT_NAME] [BATCH_SIZE]
```
Where:
  - `[SPLIT_NAME]` is the folder that contains all aligned outputs
    - main function will be applied to all files in the folder ending in `_aligned_mats.npy`
  - `[BATCH_SIZE]` is how many pairs to process at once.  

**OR**  

*7b.)* Use a different script for only precalculating the equilibrium distributions: `precalculate_equl_dists_for_pairHMM.py`

```
python precalculate_equl_dists_for_pairHMM.py [SPLIT_NAME] [BATCH_SIZE]
```
Where `[SPLIT_NAME]` and `[BATCH_SIZE]` are same as before

**Highly** recommended to do this on a GPU machine.  

#### 8.) [OPTIONAL] if splits were created in parallel in step 5a, bring all the parts back together by running-
```
python pfam_pair_processing.py \
  -task concat_parts \
  -splitname [SPLITNAME] \
  -alphabet_size [default=20]
```
on all splits.


## Example data
Example seed alignment file and trees found in `./EXAMPLE_INPUTS`. 

## Ordering
Possible emissions are indexed in alphabetical order, according to one-letter abbreviation. For a 20-element equilibrium distribution vector over amino acids:
- i=0: Alanine
- i=1: Cysteine
- i=2: Aspartic Acid (Aspartate)
- and so on

I understand this ordering is controversial. Oh well, it's too late to go back.  

Possible transitions are indexed in this order: Match, Insert, Delete.
