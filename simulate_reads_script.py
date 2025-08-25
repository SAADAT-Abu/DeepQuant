#!/usr/bin/env python3
"""
FASTQ Read Simulator with Ground Truth Tracking
Generates realistic sequencing reads from a transcriptome FASTA file
"""

import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import sys
from collections import defaultdict
import pandas as pd

def load_transcriptome(fasta_path):
    """Load transcriptome and return dict of {transcript_id: sequence}"""
    transcripts = {}
    print(f"Loading transcriptome from {fasta_path}...")
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        # Extract transcript ID (first part before |)
        transcript_id = record.id.split('|')[0]
        sequence = str(record.seq).upper()
        
        # Only keep transcripts longer than read length
        if len(sequence) >= 150:
            transcripts[transcript_id] = sequence
    
    print(f"Loaded {len(transcripts)} transcripts")
    return transcripts

def generate_zipf_abundances(n_transcripts, alpha=1.5, min_abundance=1):
    """Generate realistic transcript abundances following Zipf distribution"""
    ranks = np.arange(1, n_transcripts + 1)
    abundances = ranks ** (-alpha)
    
    # Normalize and scale
    abundances = abundances / np.sum(abundances)
    abundances = abundances * (1 - min_abundance) + min_abundance / n_transcripts
    
    return abundances

def add_sequencing_errors(sequence, error_rate=0.01, quality_scores=None):
    """Add realistic sequencing errors to a read"""
    bases = ['A', 'T', 'G', 'C']
    read_list = list(sequence)
    
    for i in range(len(read_list)):
        if random.random() < error_rate:
            # Different error types
            error_type = random.random()
            if error_type < 0.7:  # Substitution (most common)
                if read_list[i] in bases:
                    available_bases = [b for b in bases if b != read_list[i]]
                    read_list[i] = random.choice(available_bases)
            elif error_type < 0.85:  # Insertion
                read_list[i] = read_list[i] + random.choice(bases)
            # Note: Deletions are harder to simulate in fixed-length reads
    
    return ''.join(read_list)

def generate_quality_scores(read_length, mean_quality=30):
    """Generate realistic Phred quality scores"""
    # Simulate quality degradation towards 3' end
    base_qualities = np.random.normal(mean_quality, 5, read_length)
    
    # Add 3' end degradation
    degradation = np.linspace(0, 8, read_length)
    base_qualities = base_qualities - degradation
    
    # Clip to reasonable range
    base_qualities = np.clip(base_qualities, 10, 40).astype(int)
    
    # Convert to ASCII (Phred+33)
    quality_string = ''.join([chr(q + 33) for q in base_qualities])
    
    return quality_string

def simulate_reads(transcripts, num_reads=10000, read_length=150, 
                  error_rate=0.01, zipf_alpha=1.5, random_seed=42):
    """
    Simulate FASTQ reads with ground truth tracking
    
    Returns:
        reads: List of (read_id, sequence, quality) tuples
        ground_truth: DataFrame with read mapping information
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"Simulating {num_reads} reads of length {read_length}...")
    
    transcript_ids = list(transcripts.keys())
    transcript_sequences = [transcripts[tid] for tid in transcript_ids]
    
    # Generate realistic abundances
    abundances = generate_zipf_abundances(len(transcript_ids), zipf_alpha)
    
    # Convert abundances to read counts
    read_counts = np.random.multinomial(num_reads, abundances)
    
    reads = []
    ground_truth_data = []
    read_counter = 0
    
    print("Abundance distribution:")
    top_10_indices = np.argsort(abundances)[-10:][::-1]
    for idx in top_10_indices:
        print(f"  {transcript_ids[idx]}: {read_counts[idx]} reads ({abundances[idx]*100:.2f}%)")
    
    for transcript_idx, (tid, sequence) in enumerate(zip(transcript_ids, transcript_sequences)):
        reads_to_generate = read_counts[transcript_idx]
        
        if reads_to_generate == 0:
            continue
        
        for read_num in range(reads_to_generate):
            read_counter += 1
            
            # Random start position
            max_start = len(sequence) - read_length
            if max_start <= 0:
                continue
            
            start_pos = random.randint(0, max_start)
            end_pos = start_pos + read_length
            
            # Extract read sequence
            read_seq = sequence[start_pos:end_pos]
            
            # Add sequencing errors
            read_seq_with_errors = add_sequencing_errors(read_seq, error_rate)
            
            # Trim/pad to exact length if necessary
            if len(read_seq_with_errors) > read_length:
                read_seq_with_errors = read_seq_with_errors[:read_length]
            elif len(read_seq_with_errors) < read_length:
                # Pad with A's if somehow shorter
                read_seq_with_errors += 'A' * (read_length - len(read_seq_with_errors))
            
            # Generate quality scores
            quality_scores = generate_quality_scores(read_length)
            
            # Create read ID
            read_id = f"read_{read_counter:06d}"
            
            reads.append((read_id, read_seq_with_errors, quality_scores))
            
            # Track ground truth
            ground_truth_data.append({
                'read_id': read_id,
                'true_transcript': tid,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'original_sequence': read_seq,
                'errors_added': read_seq != read_seq_with_errors,
                'transcript_length': len(sequence)
            })
        
        if transcript_idx % 100 == 0:
            print(f"  Processed {transcript_idx}/{len(transcript_ids)} transcripts...")
    
    print(f"Generated {len(reads)} reads from {len([c for c in read_counts if c > 0])} transcripts")
    
    # Create ground truth DataFrame
    ground_truth_df = pd.DataFrame(ground_truth_data)
    
    return reads, ground_truth_df

def write_fastq(reads, output_path):
    """Write reads to FASTQ file"""
    print(f"Writing FASTQ file to {output_path}...")
    
    with open(output_path, 'w') as f:
        for read_id, sequence, quality in reads:
            f.write(f"@{read_id}\n")
            f.write(f"{sequence}\n")
            f.write(f"+\n")
            f.write(f"{quality}\n")
    
    print(f"FASTQ file written with {len(reads)} reads")

def write_ground_truth(ground_truth_df, output_path):
    """Write ground truth mapping to CSV"""
    print(f"Writing ground truth to {output_path}...")
    ground_truth_df.to_csv(output_path, index=False)
    print(f"Ground truth written with {len(ground_truth_df)} entries")

def write_abundance_truth(ground_truth_df, transcripts, output_path):
    """Write true transcript abundances"""
    print(f"Writing abundance ground truth to {output_path}...")
    
    # Count reads per transcript
    abundance_counts = ground_truth_df['true_transcript'].value_counts()
    
    # Create abundance DataFrame
    abundance_data = []
    for transcript_id in transcripts.keys():
        count = abundance_counts.get(transcript_id, 0)
        tpm = (count / len(ground_truth_df)) * 1e6  # Simple TPM calculation
        
        abundance_data.append({
            'transcript_id': transcript_id,
            'read_count': count,
            'tpm': tpm,
            'transcript_length': len(transcripts[transcript_id])
        })
    
    abundance_df = pd.DataFrame(abundance_data)
    abundance_df = abundance_df.sort_values('read_count', ascending=False)
    abundance_df.to_csv(output_path, index=False)
    
    print(f"Abundance ground truth written for {len(abundance_df)} transcripts")
    return abundance_df

def main():
    parser = argparse.ArgumentParser(description='Simulate FASTQ reads with ground truth')
    parser.add_argument('input_fasta', help='Input transcriptome FASTA file')
    parser.add_argument('-n', '--num-reads', type=int, default=10000, 
                        help='Number of reads to generate (default: 10000)')
    parser.add_argument('-l', '--read-length', type=int, default=150,
                        help='Read length (default: 150)')
    parser.add_argument('-e', '--error-rate', type=float, default=0.01,
                        help='Sequencing error rate (default: 0.01)')
    parser.add_argument('-a', '--zipf-alpha', type=float, default=1.5,
                        help='Zipf distribution parameter for abundances (default: 1.5)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('-o', '--output-prefix', default='simulated',
                        help='Output file prefix (default: simulated)')
    
    args = parser.parse_args()
    
    # Load transcriptome
    transcripts = load_transcriptome(args.input_fasta)
    
    if len(transcripts) == 0:
        print("Error: No valid transcripts found!")
        sys.exit(1)
    
    # Simulate reads
    reads, ground_truth_df = simulate_reads(
        transcripts=transcripts,
        num_reads=args.num_reads,
        read_length=args.read_length,
        error_rate=args.error_rate,
        zipf_alpha=args.zipf_alpha,
        random_seed=args.seed
    )
    
    # Write outputs
    fastq_file = f"{args.output_prefix}_reads.fastq"
    truth_file = f"{args.output_prefix}_ground_truth.csv"
    abundance_file = f"{args.output_prefix}_abundances.csv"
    
    write_fastq(reads, fastq_file)
    write_ground_truth(ground_truth_df, truth_file)
    abundance_df = write_abundance_truth(ground_truth_df, transcripts, abundance_file)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Total reads generated: {len(reads)}")
    print(f"Transcripts with reads: {len(abundance_df[abundance_df['read_count'] > 0])}")
    print(f"Most abundant transcript: {abundance_df.iloc[0]['transcript_id']} ({abundance_df.iloc[0]['read_count']} reads)")
    print(f"Error rate applied: {args.error_rate}")
    print(f"Average read quality: ~30")
    
    print(f"\nOutput files:")
    print(f"  FASTQ: {fastq_file}")
    print(f"  Ground truth mapping: {truth_file}")
    print(f"  True abundances: {abundance_file}")
    
    print(f"\nTop 10 most abundant transcripts:")
    for i, row in abundance_df.head(10).iterrows():
        print(f"  {row['transcript_id']}: {row['read_count']} reads ({row['tpm']:.1f} TPM)")

if __name__ == "__main__":
    main()