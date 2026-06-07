"""
ESM_predict_mutation_effects.py

Developed by Kourosh Salehi-Ashtiani and ChatGPT o1, Oct 11, 2024.
Modified by Claude 3.5 for multi-GPU support and bug fixes, Oct 12, 2024.

Predicts the effect of every possible single amino-acid substitution in each
input protein sequence using an ESM-1v model, scoring each mutation by its
delta log-probability relative to the wild-type residue.

Requirements:
    Create a Python 3.10 conda env, e.g.:
        conda create -n esm python=3.10
        conda activate esm
        pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
        pip install pandas fair-esm biopython
        conda install tqdm

Usage:
    # Predict mutation effects for all amino-acid positions:
    python ESM_predict_mutation_effects.py input_sequences.fasta output_predictions.csv --cuda

    # Use multiple GPUs:
    python ESM_predict_mutation_effects.py input_sequences.fasta output_predictions.csv --cuda --num_gpus 2
"""

import os
import csv
import torch
import esm
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import argparse
import warnings

def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def generate_mutations(sequence, positions=None):
    """Generate all possible single amino acid mutations for a given sequence."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    mutations = []
    seq_list = list(sequence)
    seq_length = len(sequence)

    if positions is None:
        positions = range(1, seq_length + 1)

    for idx in positions:
        original_aa = sequence[idx - 1]
        for aa in amino_acids:
            if aa != original_aa:
                mutated_seq = seq_list.copy()
                mutated_seq[idx - 1] = aa
                mutated_sequence = "".join(mutated_seq)
                mutations.append({
                    'mutation': f"{original_aa}{idx}{aa}",
                    'position': idx,
                    'original_aa': original_aa,
                    'mutated_aa': aa,
                    'mutated_sequence': mutated_sequence
                })
    return mutations

def predict_mutation_effects(wt_sequence, mutations, model, alphabet, device):
    """Predict the effects of mutations using the ESM-1v model."""
    batch_converter = alphabet.get_batch_converter()
    results = []
    batch_size = 64  # Adjust based on your GPU memory

    for i in tqdm(range(0, len(mutations), batch_size), desc="Predicting mutations"):
        batch = mutations[i:i+batch_size]
        sequences = [(mut['mutation'], mut['mutated_sequence']) for mut in batch]
        labels, strs, tokens = batch_converter(sequences)
        tokens = tokens.to(device)

        with torch.no_grad():
            logits = model(tokens)["logits"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        for j, mut in enumerate(batch):
            position = mut['position'] - 1
            original_aa = mut['original_aa']
            mutated_aa = mut['mutated_aa']

            token_idx = position + 1
            wt_token = alphabet.get_idx(original_aa)
            mut_token = alphabet.get_idx(mutated_aa)

            wt_log_prob = log_probs[j, token_idx, wt_token].item()
            mut_log_prob = log_probs[j, token_idx, mut_token].item()
            delta_log_prob = mut_log_prob - wt_log_prob

            results.append({
                'mutation': f"{original_aa}{mut['position']}{mutated_aa}",
                'position': mut['position'],
                'original_aa': original_aa,
                'mutated_aa': mutated_aa,
                'delta_log_prob': delta_log_prob
            })
    return results

def main():
    warnings.filterwarnings(
        "ignore",
        message=".*Regression weights not found, predicting contacts will not produce correct results.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(description="Predict mutation effects using ESM-1v.")
    parser.add_argument("input_fasta", help="Input FASTA file containing protein sequences.")
    parser.add_argument("output_csv", help="Output CSV file to save mutation predictions.")
    parser.add_argument("--positions", nargs='+', type=int, default=None,
                        help="Specific positions to mutate (1-based indexing). If not provided, all positions are mutated.")
    parser.add_argument("--model", type=int, default=1, choices=[1,2,3,4,5],
                        help="ESM-1v model number (1-5). Higher numbers may give better performance.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computation.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use.")
    args = parser.parse_args()

    # Check CUDA availability and set device
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.cuda = False
        args.num_gpus = 0

    device = torch.device("cuda" if args.cuda else "cpu")

    # Load the specified ESM-1v model
    model_name = f"esm1v_t33_650M_UR90S_{args.model}"
    print(f"Loading model {model_name}...")
    model, alphabet = esm.pretrained.__dict__[model_name]()

    if args.cuda:
        if args.num_gpus > 1:
            print(f"Using {args.num_gpus} GPUs")
            model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
        model = model.to(device)
    model.eval()

    # Load sequences
    sequences = load_sequences(args.input_fasta)

    # Process each sequence
    for seq_id, wt_sequence in sequences:
        print(f"Processing sequence {seq_id}...")
        mutations = generate_mutations(wt_sequence, positions=args.positions)

        # Predict mutation effects
        results = predict_mutation_effects(wt_sequence, mutations, model, alphabet, device)

        # Create a DataFrame for this sequence
        df = pd.DataFrame(results)
        df.columns = [f"{seq_id}_{col}" for col in df.columns]
        df.reset_index(drop=True, inplace=True)

        # Write out the output after processing each protein
        if os.path.exists(args.output_csv):
            existing_df = pd.read_csv(args.output_csv)
            existing_df.reset_index(drop=True, inplace=True)
            combined_df = pd.concat([existing_df, df], axis=1)
        else:
            combined_df = df

        combined_df.to_csv(args.output_csv, index=False)
        print(f"Results for {seq_id} have been written to {args.output_csv}")

if __name__ == "__main__":
    main()
