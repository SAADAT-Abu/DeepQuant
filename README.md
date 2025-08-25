# DeepQuant

## Overview

DeepQuant (v2) is a transcriptome quantification tool that uses deep learning and gene family relationships to accurately assign RNA sequencing reads to their transcript origins. Unlike traditional alignment-based methods, DeepQuant V2 uses transformer architectures with hierarchical modeling to understand both gene-level similarities and isoform-specific differences.

## Our Approach

### Core Innovation: Gene-Aware Hierarchical Learning

DeepQuant V2 introduces a two-level hierarchical approach:

1. **Gene-Level Understanding**: First learns to identify which gene family a read belongs to
2. **Isoform-Level Discrimination**: Then distinguishes between specific isoforms within that gene family

### Technical Architecture

#### 1. Multi-Level Transformer Encoding
- **Base Encoder**: Pre-trained DNABERT-2 model for foundational DNA sequence understanding
- **Gene Encoder**: Captures common patterns shared across isoforms of the same gene
- **Isoform Encoder**: Learns discriminative features specific to individual transcripts
- **Multi-Scale Attention**: Processes both local sequence motifs and global sequence context

#### 2. Gene-Aware Contrastive Learning
- **Dual-Level Training**: Simultaneously learns gene family relationships and isoform distinctions
- **Contrastive Loss**: Pulls together similar sequences (same gene/isoform) and pushes apart dissimilar ones
- **Hierarchical Supervision**: Uses both gene-level and transcript-level labels during training

#### 3. Uncertainty-Aware Assignment
- **Confidence Scoring**: Each assignment comes with a confidence estimate
- **Adaptive Thresholds**: Similarity thresholds adjust based on gene complexity and uncertainty
- **Multiple Assignment Strategies**: Hierarchical, joint, and ensemble approaches for different scenarios

### Methodological Workflow

1. **Data Preparation**: Parse transcriptome FASTA and organize transcripts by gene families
2. **Model Training**: Train hierarchical encoder using gene-aware contrastive learning
3. **Index Building**: Create searchable vector database of all transcript embeddings
4. **Read Processing**: Convert each sequencing read into gene and isoform embeddings
5. **Hierarchical Search**: First find candidate genes, then best-matching isoforms within those genes
6. **Confident Assignment**: Use uncertainty-aware strategies to assign reads to transcripts
7. **Quantification**: Aggregate assignments to produce transcript abundance estimates

### **This script is currently under active development and should be considered experimental.** 
