"""
Deep Learning-powered Transcriptome Quantification Tool with Pretrained Models
Using pretrained genomic BERT models for faster development and better performance
Supported pretrained models:
- DNABERT: https://github.com/jerryji1993/DNABERT
- DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2  
- Nucleotide Transformer: https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-human-ref
- GenomicBERT: Various genomic BERT variants on HuggingFace
Key advantages:
- Skip training from scratch
- Better sequence understanding from large-scale pretraining
- Transfer learning for specific tasks
- Faster development cycle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import faiss
from collections import defaultdict
import time
import os
import tempfile
from typing import Dict, List, Tuple, Optional
import logging
import pandas as pd
# Configure logging to display information about the script's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GenomicTokenizer:
    """
    A wrapper class for various genomic tokenization strategies.
    
    This class handles the tokenization of DNA sequences for different
    pretrained models DNABERT-2.
    """
    
    def __init__(self, model_name: str, kmer_size: int = 6):
        """
        Initializes the tokenizer based on the specified model name.
        Args:
            model_name (str): The name of the pretrained model. 
                              Supported: "DNABERT", "DNABERT2", "NucleotideTransformer".
            kmer_size (int): The k-mer size, primarily used for the "DNABERT" model.
        """
        self.model_name = model_name
        self.kmer_size = kmer_size
        
        # Load the tokenizer from HuggingFace
        if model_name == "DNABERT2":
            self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    def _create_dnabert_vocab(self, k: int) -> str:
        """
        Generates a temporary vocabulary file for a k-mer based DNABERT model.
        Args:
            k (int): The size of the k-mers.
        Returns:
            str: The file path to the generated vocabulary file.
        """
        bases = ['A', 'T', 'G', 'C']
        
        def generate_kmers(length, current=""):
            if length == 0:
                return [current]
            result = []
            for base in bases:
                result.extend(generate_kmers(length - 1, current + base))
            return result
        
        kmers = generate_kmers(k)
        
        # Define special tokens required by BERT
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        vocab_content = special_tokens + kmers
        
        # Write the vocabulary to a temporary file
        fd, vocab_path = tempfile.mkstemp(suffix='.txt', text=True)
        with os.fdopen(fd, 'w') as f:
            for token in vocab_content:
                f.write(f"{token}\n")
        
        return vocab_path
    def tokenize_sequence(self, sequence: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a single DNA sequence according to the model's requirements.
        Args:
            sequence (str): The DNA sequence to tokenize.
            max_length (int): The maximum sequence length for padding/truncation.
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing tokenized 'input_ids', 
                                     'attention_mask', etc., as PyTorch tensors.
        """
        # Pre-process sequence: convert to uppercase and replace 'N' characters
        sequence = sequence.upper().replace('N', 'A')
        
        if self.model_name == "DNABERT2":
            # DNABERT2 and NucleotideTransformer can process raw DNA sequences directly
            return self.tokenizer(
                sequence,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

class PretrainedGenomicEncoder(nn.Module):
    """
    An encoder that uses a pretrained genomic model to generate sequence embeddings.
    
    This module wraps a HuggingFace transformer model and adds a projection
    head to produce standardized, L2-normalized embeddings suitable for
    similarity search.
    """
    
    def __init__(self, model_name: str, fine_tune: bool, embedding_dim: int):
        super().__init__()
        self.model_name = model_name
        self.fine_tune = fine_tune
        
        # Load the specified pretrained model from HuggingFace
        if model_name == "DNABERT2":
            model_path = "zhihan1996/DNABERT-2-117M"
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden_size = self.encoder.config.hidden_size
        
        # Freeze the encoder parameters if not fine-tuning
        if not fine_tune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection head to map the model's hidden size to the desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate sequence embeddings.
        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
        Returns:
            torch.Tensor: A batch of L2-normalized embeddings.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the sequence representation. Use mean pooling as a robust default.
        last_hidden_state = outputs.last_hidden_state
        # Multiply by attention mask to exclude padding tokens from the mean calculation
        masked_hidden_state = last_hidden_state * attention_mask.unsqueeze(-1)
        # Sum over the sequence length and divide by the number of non-padded tokens
        summed_hidden_state = masked_hidden_state.sum(dim=1)
        summed_mask = attention_mask.sum(dim=1, keepdim=True)
        sequence_embedding = summed_hidden_state / summed_mask
        
        # Pass through the projection head
        projected = self.projection(sequence_embedding)
        
        # L2 normalize the final embedding for use with cosine similarity
        return F.normalize(projected, p=2, dim=-1)

class PretrainedDLQuantifier:
    """
    A tool for transcriptome quantification using pretrained deep learning models.
    This class provides a complete pipeline:
    1. Fine-tuning the model on a specific transcriptome.
    2. Building a searchable index of transcript embeddings.
    3. Quantifying sequencing reads by mapping them to transcripts.
    """
    def __init__(self, model_name: str, kmer_size: int = 6, fine_tune: bool = True, embedding_dim: int = 256):
        """
        Initializes the quantifier with a specified model and parameters.
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = GenomicTokenizer(model_name, kmer_size)
        self.model = PretrainedGenomicEncoder(model_name, fine_tune, embedding_dim)
        self.model.to(self.device)
        
        # Initialize attributes for the FAISS index
        self.transcript_index = None
        self.transcript_embeddings = None
        self.transcript_ids = None
        
        logger.info(f"Initialized quantifier with model: {model_name} on device: {self.device}")
    def _tokenize_batch(self, sequences: List[str], max_length: int = 150) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a batch of sequences efficiently.
        """
        # The tokenizer for DNABERT2 is highly optimized for batch processing
        tokenized = self.tokenizer.tokenizer(
            sequences,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Ensure only model-compatible arguments are returned (e.g., input_ids, attention_mask)
        # Some tokenizers might add extra keys like 'token_type_ids' which are not needed by all models
        model_args = {'input_ids', 'attention_mask'}
        filtered_tokens = {k: v.to(self.device) for k, v in tokenized.items() if k in model_args}
        
        return filtered_tokens
    def build_index(self, transcripts: Dict[str, str], batch_size: int = 12):
        """
        Builds a FAISS search index from transcript embeddings.
        Args:
            transcripts (Dict[str, str]): A dictionary mapping transcript IDs to sequences.
            batch_size (int): The number of transcripts to process in each batch.
        """
        logger.info("Building transcript index from embeddings...")
        self.model.eval()
        self.transcript_ids = list(transcripts.keys())
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.transcript_ids), batch_size):
                batch_ids = self.transcript_ids[i:i+batch_size]
                batch_seqs = [transcripts[tid] for tid in batch_ids]
                
                tokens = self._tokenize_batch(batch_seqs)
                batch_embeddings = self.model(**tokens)
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        self.transcript_embeddings = np.vstack(all_embeddings)
        
        # Create a FAISS index for efficient similarity search (Inner Product is equivalent to Cosine Similarity for normalized vectors)
        embedding_dim = self.transcript_embeddings.shape[1]
        self.transcript_index = faiss.IndexFlatIP(embedding_dim)
        self.transcript_index.add(self.transcript_embeddings.astype('float32'))
        
        logger.info(f"Successfully built index with {self.transcript_index.ntotal} transcripts.")
    def quantify_reads(self, reads: List[str], batch_size: int = 32, top_k: int = 10, similarity_threshold: float = 0.9) -> Dict[str, float]:
        """
        Quantifies transcript abundance from a list of sequencing reads.
        Args:
            reads (List[str]): A list of DNA sequences from reads.
            batch_size (int): The number of reads to process in each batch.
            top_k (int): The number of top matching transcripts to consider for each read.
            similarity_threshold (float): The minimum similarity score for a read-transcript match.
        Returns:
            Dict[str, float]: A dictionary mapping transcript IDs to their estimated counts.
        """
        if self.transcript_index is None:
            raise ValueError("Index has not been built. Please call build_index() first.")
        
        logger.info(f"Quantifying {len(reads)} reads...")
        self.model.eval()
        transcript_counts = defaultdict(float)
        with torch.no_grad():
            for i in range(0, len(reads), batch_size):
                batch_reads = reads[i:i+batch_size]
                tokens = self._tokenize_batch(batch_reads)
                read_embeddings = self.model(**tokens).cpu().numpy()
                
                # Search the FAISS index for the top_k most similar transcripts
                similarities, indices = self.transcript_index.search(read_embeddings.astype('float32'), top_k)
                
                # Assign reads to transcripts using a probabilistic model
                for j in range(len(batch_reads)):
                    read_sims = similarities[j]
                    read_indices = indices[j]
                    
                    # Filter matches below the similarity threshold
                    valid_mask = read_sims > similarity_threshold
                    if not np.any(valid_mask):
                        continue
                    
                    valid_sims = read_sims[valid_mask]
                    valid_indices = read_indices[valid_mask]
                    
                    # Use softmax with temperature scaling for probabilistic assignment
                    # This assigns fractions of a read's count to multiple transcripts if they are similarly good matches.
                    temperature = 0.2 # Lower temp -> more confident assignments
                    exp_sims = np.exp(valid_sims / temperature)
                    probabilities = exp_sims / exp_sims.sum()
                    
                    for idx, prob in zip(valid_indices, probabilities):
                        transcript_id = self.transcript_ids[idx]
                        transcript_counts[transcript_id] += prob
        
        return dict(transcript_counts)
    def save_model(self, path: str):
        """Saves the model state and index data to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'transcript_ids': self.transcript_ids,
            'transcript_embeddings': self.transcript_embeddings
        }, path)
        logger.info(f"Model and index data saved to {path}")
    def load_model(self, path: str):
        """Loads a model and rebuilds the FAISS index from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.transcript_ids = checkpoint['transcript_ids']
        self.transcript_embeddings = checkpoint['transcript_embeddings']
        
        if self.transcript_embeddings is not None:
            d = self.transcript_embeddings.shape[1]
            self.transcript_index = faiss.IndexFlatIP(d)
            self.transcript_index.add(self.transcript_embeddings.astype('float32'))
        
        logger.info(f"Model and index data loaded from {path}")

def load_transcriptome(fasta_path: str) -> Dict[str, str]:
    """
    Loads a transcriptome from a FASTA file, parsing GENCODE-style headers.
    Args:
        fasta_path (str): Path to the transcriptome FASTA file.
    Returns:
        Dict[str, str]: A dictionary mapping transcript IDs to their sequences.
    """
    transcripts = {}
    transcript_to_gene = {} 
    logger.info(f"Loading transcriptome from {fasta_path}...")
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Example: >ENST...|ENSG...|...|...|...|GENE_NAME|...
        fields = record.id.split('|')
        transcript_id = fields[0]  # ENST...
        gene_name = fields[5]  # GENE_NAME
        transcripts[transcript_id] = str(record.seq)
        transcript_to_gene[transcript_id] = gene_name
    logger.info(f"Loaded {len(transcripts)} transcripts.")
    return transcripts, transcript_to_gene
def load_reads(fastq_path: str, max_reads: Optional[int] = None) -> List[str]:
    """
    Loads sequencing reads from a FASTQ file.
    Args:
        fastq_path (str): Path to the FASTQ file.
        max_reads (int, optional): Maximum number of reads to load. Defaults to None (load all).
    Returns:
        List[str]: A list of read sequences.
    """
    reads = []
    logger.info(f"Loading reads from {fastq_path}...")
    for i, record in enumerate(SeqIO.parse(fastq_path, "fastq")):
        if max_reads and i >= max_reads:
            break
        reads.append(str(record.seq))
    logger.info(f"Loaded {len(reads)} reads.")
    return reads

if __name__ == "__main__":
    # --- Configuration ---
    # Useing DNABERT-2
    CHOSEN_MODEL = "DNABERT2" 
    
    # Paths to your data files
    # Make sure to replace these with the actual paths to your files
    TRANSCRIPTOME_PATH = "/run/media/saadat/A/tools/DeepQuant/gencode.v47.transcripts_1000.fa"
    READS_PATH = "/run/media/saadat/A/tools/DeepQuant/simulated_reads.fastq"
    # Path to save/load the trained model
    MODEL_SAVE_PATH = "transcript_quantifier.pth"
    
    # Output path for the quantification results
    RESULTS_PATH = "quantification_results.csv"
    
    # --- Step 1: Initialize the Quantifier ---
    quantifier = PretrainedDLQuantifier(
        model_name=CHOSEN_MODEL,
        fine_tune=True, # Set to True if you plan to fine-tune
        embedding_dim=256
    )
    
    # --- Step 2: Load Data ---
    # Load all transcripts and transcript-to-gene mapping
    transcripts, transcript_to_gene = load_transcriptome(TRANSCRIPTOME_PATH)
    # Load all reads (no max_reads limit)
    reads = load_reads(READS_PATH)
    
    # --- Step 3: Build the Transcript Index ---
    # This step is required before quantification.
    # It embeds all reference transcripts and builds the search index.
    start_time = time.time()
    quantifier.build_index(transcripts)
    end_time = time.time()
    logger.info(f"Index building took {end_time - start_time:.2f} seconds.")
    
    # --- Step 4: Quantify Reads ---
    start_time = time.time()
    counts = quantifier.quantify_reads(reads)
    end_time = time.time()
    logger.info(f"Quantification of {len(reads)} reads took {end_time - start_time:.2f} seconds.")
    
    # --- Step 5: Save Results ---
    if counts:
        results_df = pd.DataFrame([
            {
                'transcript_id': tid,
                'gene_name': transcript_to_gene.get(tid, "NA"),
                'estimated_count': float(count)
            }
            for tid, count in counts.items()
        ])
        results_df = results_df.sort_values(by='estimated_count', ascending=False)
        results_df.to_csv(RESULTS_PATH, index=False)
        logger.info(f"Results saved to {RESULTS_PATH}")
        print("\n--- Top 10 Expressed Transcripts ---")
        print(results_df.head(10))
    else:
        logger.warning("No reads were quantified. Check similarity threshold or model performance.")
        
    # --- Optional: Save the Model for Future Use ---
    # This saves the model's state and the transcript index data
    # quantifier.save_model(MODEL_SAVE_PATH)
    logger.info("Script finished successfully.")