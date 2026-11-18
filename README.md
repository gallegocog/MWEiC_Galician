# MWEiC: Multi-Word Expressions in Context - GL

This repository contains the MWEiC (Multi-Word Expressions in Context) dataset, which is used to explore contextual meaning variation in multiword expressions (MWEs). We adopted the rationale of the Word-in-Context (WiC) task, in which a classifier must determine whether a word occurring in two contexts has the same meaning, and applied it to MWEs.


## Datasets
Using [noun-adjective](https://github.com/Castro-L/MWE_dataset_gl) and [verb-object](https://github.com/marcospln/Galician_MWE_dataset) MWE datasets for Galician, which include potentially different meanings and varying degrees of semantic compositionality, we created WiC-like datasets for each syntactic pattern independently. Specifically, we created two versions of the dataset for each pattern:

- **SentDisjCorp** (Sentence Disjoint): This version adopts a strict constraint whereby each sentence appears in at most one instance across the entire dataset.

- **MWEGenCorp** (MWE Generalization): In this case, a sentence may appear in multiple pairs within the same split (e.g., train), but it is prohibited from appearing in more than one split.

In each of the resulting four datasets, every instance consists of an MWE and two contexts, labeled according to whether the MWE has the same meaning in both sentences.

These datasets, together with details about their contents, can be found in [`/datasets`](./datasets/).


## Scripts
We conducted an extensive benchmarking study using standard WiC evaluation methods to assess our proposed MWEiC task. Specifically, we evaluated supervised classifiers initialized with different types of composite vectors derived from LMs, including encoder and generative models of varying sizes. We further investigated performance across different dataset settings (e.g., noun-adjective vs verb-object MWEs) and analyzed how it varies across model layers. We also compared these classifiers with static word vectors and with prompt-based strategies using instruction-tuned models.

The scripts used to conduct these experiments can be found in [`/scripts`](./scripts/).