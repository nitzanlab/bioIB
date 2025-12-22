# bioIB
bioIB package for scRNA-seq analysis

This repository provides the code for bioIB, described in our article - 
"bioIB: Identifying informative metagenes in single-cell RNA-seq with the Information Bottleneck algorithm"

![intro_figure](https://github.com/user-attachments/assets/ffe67ad3-1ef0-4d14-aec0-7ae6184c0199)

bioIB is a novel method for scRNA-seq analysis, based on the Information Bottleneck algorithm. 
Given the cellular labels representing the signal of interest, such as cell type, bioIB generates probabilistic gene clusters, or metagenes, that capture the major biological processes differentially activated between the defined cellular populations.

# This repository

This repository contains:

1. The bioIB package code.
2. Jupyter notebook tutorial exemplifying the use of the bioIB package functions.

# Installation
For running on CPU (sufficient for flat clustering and hierarchical clustering of ~500-1000 genes):

```
pip install jax
pip install bioib
```
For running on GPU on Linux (preferable for hierarchical clustering of >1000 genes):

```
pip install "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install bioib
```

