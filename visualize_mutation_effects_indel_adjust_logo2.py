"""
visualize_mutation_effects_indel_adjust_logo2.py

Developed by Kourosh Salehi-Ashtiani and ChatGPT o1, Oct 12, 2024.

Visualizes the mutation effects predicted by ESM_predict_mutation_effects.py.
For each sequence it generates a heatmap, a per-position boxplot, an
average-effect bar plot by amino acid, and a sequence logo.

This version adjusts for indels: it aligns each variant to the WT, so both
--wt_sequence and --variant_sequences are required. The --positions and
--ranges arguments are optional.

Requirements:
    pip install pandas matplotlib seaborn logomaker biopython

Usage:
    python visualize_mutation_effects_indel_adjust_logo2.py input_predictions.csv output_directory --wt_sequence wt_sequence.fasta --variant_sequences variant_sequences.fasta

    # Restrict to specific positions / ranges:
    python visualize_mutation_effects_indel_adjust_logo2.py input_predictions.csv output_directory --wt_sequence wt_sequence.fasta --variant_sequences variant_sequences.fasta --positions 33 66 --ranges 24-33
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
from Bio import SeqIO, pairwise2
import logomaker  # Import Logomaker

def load_sequences_from_csv(csv_file):
    """Load sequence IDs from the CSV file."""
    df = pd.read_csv(csv_file, nrows=1)
    seq_ids = set(col.rsplit('_', 1)[0] for col in df.columns if '_' in col)
    return seq_ids

def load_data(csv_file):
    """Load the CSV file and separate data by sequence efficiently."""
    df = pd.read_csv(csv_file)
    sequences = {}

    # Get unique sequence IDs by splitting column names at the last underscore
    seq_ids = set(col.rsplit('_', 1)[0] for col in df.columns if '_' in col)

    for seq_id in seq_ids:
        # Select columns for this sequence
        cols = [col for col in df.columns if col.startswith(f"{seq_id}_")]
        # Create DataFrame for this sequence
        seq_df = df[cols].copy()
        # Rename columns to remove sequence ID prefix and underscore
        seq_df.columns = [col[len(seq_id)+1:] for col in seq_df.columns]
        sequences[seq_id] = seq_df

    return sequences

def parse_positions(position_args, range_args):
    """Parse position and range arguments to create a set of positions."""
    positions = set()

    # Add individual positions
    if position_args:
        for pos_str in position_args:
            try:
                pos = int(pos_str)
                positions.add(pos)
            except ValueError:
                print(f"Warning: Invalid position '{pos_str}' skipped.")

    # Add ranges of positions
    if range_args:
        for range_str in range_args:
            try:
                start_str, end_str = range_str.split('-')
                start = int(start_str)
                end = int(end_str)
                if start <= end:
                    positions.update(range(start, end + 1))
                else:
                    print(f"Warning: Invalid range '{range_str}' skipped.")
            except ValueError:
                print(f"Warning: Invalid range '{range_str}' skipped.")

    return positions

def align_sequences(wt_sequence, variant_sequence):
    """Align WT sequence with variant sequence and create position mapping."""
    # Perform global alignment
    alignments = pairwise2.align.globalxx(wt_sequence, variant_sequence)
    best_alignment = alignments[0]
    aligned_wt_seq = best_alignment.seqA
    aligned_var_seq = best_alignment.seqB

    # Create mapping from WT positions to variant positions
    wt_pos = 0
    var_pos = 0
    position_mapping = {}

    for wt_aa, var_aa in zip(aligned_wt_seq, aligned_var_seq):
        if wt_aa != '-':
            wt_pos += 1
        if var_aa != '-':
            var_pos += 1
        if wt_aa != '-' and var_aa != '-':
            position_mapping[wt_pos] = var_pos
        elif wt_aa != '-' and var_aa == '-':
            # Deletion in variant
            position_mapping[wt_pos] = None  # No corresponding position
        elif wt_aa == '-' and var_aa != '-':
            # Insertion in variant (skip mapping)
            pass
    return position_mapping

def plot_heatmap(seq_data, seq_id, output_dir):
    """Create a heatmap of mutation effects."""
    plt.figure(figsize=(20, 10))

    # Create pivot table, handling duplicate entries by taking the mean
    pivot_data = seq_data.pivot_table(index='mutated_aa', columns='wt_position', values='delta_log_prob', aggfunc='mean')

    # Determine the color scale range
    abs_max = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))

    sns.heatmap(pivot_data, cmap='RdBu_r', center=0, vmin=-abs_max, vmax=abs_max)
    plt.title(f'Mutation Effects Heatmap for {seq_id}')
    plt.xlabel('WT Position')
    plt.ylabel('Mutated Amino Acid')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_heatmap.png'))
    plt.close()

def plot_position_boxplot(seq_data, seq_id, output_dir):
    """Create a boxplot of mutation effects by position."""
    plt.figure(figsize=(20, 10))
    sns.boxplot(x='wt_position', y='delta_log_prob', data=seq_data)
    plt.title(f'Mutation Effects by Position for {seq_id}')
    plt.xlabel('WT Position')
    plt.ylabel('Delta Log Probability')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_position_boxplot.png'))
    plt.close()

def plot_aa_barplot(seq_data, seq_id, output_dir):
    """Create a barplot of average mutation effects by amino acid."""
    plt.figure(figsize=(12, 6))
    aa_effects = seq_data.groupby('mutated_aa')['delta_log_prob'].mean().sort_values()
    sns.barplot(x=aa_effects.index, y=aa_effects.values)
    plt.title(f'Average Mutation Effects by Amino Acid for {seq_id}')
    plt.xlabel('Mutated Amino Acid')
    plt.ylabel('Average Delta Log Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{seq_id}_aa_barplot.png'))
    plt.close()

def plot_sequence_logo(seq_data, seq_id, output_dir, positions_to_plot):
    """Create a sequence logo of mutation effects."""
    print(f"Preparing data for sequence logo for {seq_id}...")
    # Create pivot table with positions as index and amino acids as columns
    logo_data = seq_data.pivot_table(
        index='wt_position',
        columns='mutated_aa',
        values='delta_log_prob',
        aggfunc='mean'
    )

    # Reindex to include all positions in positions_to_plot
    if positions_to_plot:
        all_positions = sorted(positions_to_plot)
    else:
        all_positions = sorted(seq_data['wt_position'].unique())
    logo_data = logo_data.reindex(all_positions, fill_value=0)

    # Replace NaN with zeros (should be redundant after reindexing)
    logo_data = logo_data.fillna(0)

    # Check if logo_data is empty
    if logo_data.empty:
        print(f"No data available to plot sequence logo for {seq_id}.")
        return

    # Shift data to be positive if necessary
    min_value = logo_data.min().min()
    if min_value < 0:
        logo_data += abs(min_value)
        print(f"Data shifted by {abs(min_value)} to make all values positive.")

    print(f"Plotting sequence logo for {seq_id} with {logo_data.shape[0]} positions and {logo_data.shape[1]} amino acids.")

    # Create a Logo object without shade_below_zero
    plt.figure(figsize=(max(10, 0.5 * len(all_positions)), 6))
    logo = logomaker.Logo(
        logo_data,
        color_scheme='weblogo_protein'
    )

    # Set axis labels and title
    plt.title(f'Sequence Logo for {seq_id}')
    plt.xlabel('WT Position')
    plt.ylabel('Adjusted Delta Log Probability')

    # Adjust x-axis to show positions
    logo.ax.set_xticks(range(len(all_positions)))
    logo.ax.set_xticklabels(all_positions)
    plt.xticks(rotation=90)

    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{seq_id}_sequence_logo.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Sequence logo saved to {output_path}")

def process_sequence(seq_id, seq_data, output_dir, positions_to_plot, wt_sequence, position_mapping):
    """Process a single sequence and generate its visualizations."""
    print(f"\nProcessing sequence: {seq_id}")

    # Ensure all required columns are present
    required_columns = ['position', 'mutated_aa', 'delta_log_prob']
    missing_columns = [col for col in required_columns if col not in seq_data.columns]
    if missing_columns:
        print(f"Warning: Missing required columns {missing_columns} for {seq_id}. Skipping this sequence.")
        print(f"Available columns: {seq_data.columns.tolist()}")
        return

    # Convert position to numeric type
    seq_data = seq_data.copy()
    seq_data['position'] = pd.to_numeric(seq_data['position'], errors='coerce')

    # Remove any rows with NaN values
    seq_data = seq_data.dropna(subset=required_columns)

    if seq_data.empty:
        print(f"Warning: No valid data for {seq_id} after cleaning. Skipping this sequence.")
        return

    # Convert position to integer type (after dropping NaNs)
    seq_data['position'] = seq_data['position'].astype(int)

    # Map variant positions to WT positions
    if not position_mapping:
        print(f"Warning: No position mapping available for {seq_id}. Skipping this sequence.")
        return

    # Add 'wt_position' column
    var_to_wt_mapping = {v_pos: wt_pos for wt_pos, v_pos in position_mapping.items() if v_pos is not None}
    seq_data['wt_position'] = seq_data['position'].map(var_to_wt_mapping)

    # Remove rows where 'wt_position' is NaN (positions with no mapping)
    seq_data = seq_data.dropna(subset=['wt_position'])
    seq_data['wt_position'] = seq_data['wt_position'].astype(int)

    # Display available WT positions
    available_positions = seq_data['wt_position'].unique()
    print(f"WT positions available in {seq_id}: {available_positions}")

    # Filter data for specified WT positions
    if positions_to_plot:
        seq_data = seq_data[seq_data['wt_position'].isin(positions_to_plot)]
        print(f"Number of data points after filtering for {seq_id}: {len(seq_data)}")
        if seq_data.empty:
            print(f"Warning: No data for specified positions in {seq_id}. Skipping this sequence.")
            return
    else:
        print(f"Number of data points for {seq_id}: {len(seq_data)}")

    try:
        print(f"Generating heatmap for {seq_id}...")
        plot_heatmap(seq_data, seq_id, output_dir)
        print(f"Generating position boxplot for {seq_id}...")
        plot_position_boxplot(seq_data, seq_id, output_dir)
        print(f"Generating AA barplot for {seq_id}...")
        plot_aa_barplot(seq_data, seq_id, output_dir)
        print(f"Generating sequence logo for {seq_id}...")
        plot_sequence_logo(seq_data, seq_id, output_dir, positions_to_plot)
        print(f"Completed visualizations for {seq_id}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating visualizations for {seq_id}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Visualize mutation effects predicted by ESM_predict_mutation_effects.py")
    parser.add_argument("input_csv", help="Input predictions CSV file")
    parser.add_argument("output_dir", help="Output directory for visualizations")
    parser.add_argument("--wt_sequence", required=True, help="FASTA file containing the WT sequence")
    parser.add_argument("--variant_sequences", required=True, help="FASTA file containing variant sequences")
    parser.add_argument("--positions", nargs='*', help="Specific positions to plot (e.g., 31 41 122)")
    parser.add_argument("--ranges", nargs='*', help="Ranges of positions to plot (e.g., 24-33)")

    args = parser.parse_args()

    input_csv = args.input_csv
    output_dir = args.output_dir
    wt_sequence_file = args.wt_sequence
    variant_sequences_file = args.variant_sequences
    position_args = args.positions
    range_args = args.ranges

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load WT sequence
    wt_sequences = list(SeqIO.parse(wt_sequence_file, "fasta"))
    if len(wt_sequences) != 1:
        print("Error: WT sequence file must contain exactly one sequence.")
        sys.exit(1)
    wt_sequence = str(wt_sequences[0].seq)
    print(f"Loaded WT sequence '{wt_sequences[0].id}' with length {len(wt_sequence)}")

    # Load variant sequences
    variant_sequences = {}
    for seq_record in SeqIO.parse(variant_sequences_file, "fasta"):
        variant_sequences[seq_record.id] = str(seq_record.seq)
    print(f"Loaded {len(variant_sequences)} variant sequences.")

    print("Loading data...")
    sequences = load_data(input_csv)
    print(f"Found {len(sequences)} sequences in the data.")

    # Parse positions to plot
    positions_to_plot = parse_positions(position_args, range_args)
    if positions_to_plot:
        print(f"Positions to plot: {sorted(positions_to_plot)}")
    else:
        print("Plotting all positions.")

    # Create position mappings for all variants
    wt_position_mappings = {}
    seq_ids_in_data = sequences.keys()
    for seq_id in seq_ids_in_data:
        variant_seq = variant_sequences.get(seq_id)
        if variant_seq is None:
            print(f"Warning: No sequence data available for '{seq_id}'. Skipping.")
            continue
        mapping = align_sequences(wt_sequence, variant_seq)
        wt_position_mappings[seq_id] = mapping

    for seq_id, seq_data in sequences.items():
        position_mapping = wt_position_mappings.get(seq_id)
        if position_mapping is None:
            print(f"Warning: No position mapping available for {seq_id}. Skipping this sequence.")
            continue
        process_sequence(seq_id, seq_data, output_dir, positions_to_plot, wt_sequence, position_mapping)

    print(f"Visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    main()
