"""
calculate_variant_scores_with_proxy_PCA-annoy.py

Developed by Kourosh Salehi-Ashtiani and ChatGPT o1, Oct 12, 2024.

Aligns each variant to a wild-type (or WT-proxy) sequence, scores substitutions,
insertions, and deletions to produce a cumulative detrimental score per variant,
and optionally runs PCA, cosine-similarity, and nearest-neighbor (Spotify Annoy)
analyses over the per-variant mutation-effect vectors.

Tip: instead of the true WT, use a closely related sequence as a WT proxy so the
WT's own mutation-effect vector is not all zeros.

Requirements:
    pip install biopython pandas numpy scikit-learn matplotlib annoy

Command-line arguments:
    wt_fasta            FASTA file containing the WT (or WT-proxy) sequence (exactly one sequence).
    variants_fasta      FASTA file containing the variant sequences.
    wt_mutation_data    CSV of WT delta_log_prob scores (produced by ESM_predict_mutation_effects.py).
    output_csv          Output CSV for the cumulative detrimental scores.
    --insertion_score   Penalty applied to insertions (default -1.0).
    --deletion_score    Penalty applied to deletions (default -1.0).
    --perform_pca       Run PCA, cosine-similarity, and nearest-neighbor analysis.
    --query_variant     Variant ID to use as the nearest-neighbor query
                        (defaults to the first variant).
    --num_neighbors     Number of nearest neighbors to report (default 5).

Note: when --perform_pca is used, a PCA scatter plot is displayed; close the
figure window to let the script continue and save the cosine-similarity matrix.

Usage:
    # Cumulative scores only:
    python calculate_variant_scores_with_proxy_PCA-annoy.py wt_proxy.fasta variants.fasta wt_mutation_data.csv cumulative_scores.csv

    # With PCA / cosine-similarity / nearest-neighbor analysis:
    python calculate_variant_scores_with_proxy_PCA-annoy.py wt_proxy.fasta variants.fasta wt_mutation_data.csv cumulative_scores.csv --perform_pca

    # Specify the query variant and neighbor count (WT here is a sample ID):
    python calculate_variant_scores_with_proxy_PCA-annoy.py wt_proxy.fasta variants.fasta wt_mutation_data.csv cumulative_scores.csv --perform_pca --query_variant WT --num_neighbors 5
"""
import os
import pandas as pd
import numpy as np
from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from annoy import AnnoyIndex  # Import Annoy

def load_sequences(fasta_file):
    """Load sequences from a FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

def identify_mutations(reference_sequence, target_sequence):
    """Align sequences and identify substitutions, insertions, and deletions."""
    # Perform global alignment
    alignments = pairwise2.align.globalxx(reference_sequence, target_sequence)
    best_alignment = alignments[0]
    aligned_ref, aligned_target = best_alignment.seqA, best_alignment.seqB

    mutations = []

    position = 0  # 1-based indexing
    for ref_aa, tgt_aa in zip(aligned_ref, aligned_target):
        if ref_aa == '-':
            # Insertion in target
            mutation = {
                'position': position + 0.5,  # Use half-integer to indicate between residues
                'mutation_type': 'insertion',
                'original_aa': '-',
                'mutated_aa': tgt_aa,
                'mutation': f"ins{position+0.5}_{tgt_aa}"
            }
            mutations.append(mutation)
        elif tgt_aa == '-':
            # Deletion in target
            position += 1
            mutation = {
                'position': position,
                'mutation_type': 'deletion',
                'original_aa': ref_aa,
                'mutated_aa': '-',
                'mutation': f"del{position}_{ref_aa}"
            }
            mutations.append(mutation)
        else:
            position += 1
            if ref_aa != tgt_aa:
                # Substitution
                mutation = {
                    'position': position,
                    'mutation_type': 'substitution',
                    'original_aa': ref_aa,
                    'mutated_aa': tgt_aa,
                    'mutation': f"{ref_aa}{position}{tgt_aa}"
                }
                mutations.append(mutation)
    return mutations

def assign_mutation_scores(mutations, proxy_mutation_data, insertion_score=-1.0, deletion_score=-1.0):
    """Assign scores to substitutions and indels."""
    scores = []
    for mut in mutations:
        if mut['mutation_type'] == 'substitution':
            # Get the delta_log_prob from proxy mutation data
            mutation_str = mut['mutation']
            position = mut['position']
            original_aa = mut['original_aa']
            mutated_aa = mut['mutated_aa']

            match = proxy_mutation_data[
                (proxy_mutation_data['mutation'] == mutation_str) &
                (proxy_mutation_data['position'] == position) &
                (proxy_mutation_data['original_aa'] == original_aa) &
                (proxy_mutation_data['mutated_aa'] == mutated_aa)
            ]
            if not match.empty:
                delta_log_prob = match['delta_log_prob'].values[0]
            else:
                # Handle missing data
                delta_log_prob = 0  # Assign a neutral score or adjust as needed
        elif mut['mutation_type'] == 'insertion':
            # Assign a detrimental score for insertion
            delta_log_prob = insertion_score
        elif mut['mutation_type'] == 'deletion':
            # Assign a detrimental score for deletion
            delta_log_prob = deletion_score
        scores.append(delta_log_prob)
    return scores

def calculate_cumulative_scores(reference_sequence, variants, proxy_mutation_data, insertion_score=-1.0, deletion_score=-1.0):
    """Calculate cumulative detrimental scores for variants."""
    mutations_dict = {}
    variant_scores = {}
    variant_cumulative_scores = {}

    for variant_name, variant_seq in variants.items():
        print(f"Processing variant: {variant_name}")
        # Identify mutations
        mutations = identify_mutations(reference_sequence, variant_seq)
        mutations_dict[variant_name] = mutations

        # Assign scores
        scores = assign_mutation_scores(mutations, proxy_mutation_data, insertion_score, deletion_score)
        variant_scores[variant_name] = scores

        # Calculate cumulative score
        cumulative_score = sum(scores)
        variant_cumulative_scores[variant_name] = cumulative_score
        print(f"Cumulative detrimental score for {variant_name}: {cumulative_score}")

    return variant_cumulative_scores, mutations_dict, variant_scores

def perform_pca(variant_vectors, n_components=2):
    """Perform PCA on variant mutation effect vectors."""
    # Create DataFrame from variant_vectors
    df = pd.DataFrame.from_dict(variant_vectors, orient='index')
    df.fillna(0, inplace=True)  # Fill missing mutations with zero

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    pca_df = pd.DataFrame(data=principal_components, index=df.index,
                          columns=[f'PC{i+1}' for i in range(n_components)])

    return pca_df, pca, df  # Return df for Annoy

def plot_pca_with_annoy(pca_df, query_variant, nearest_variants):
    """Plot PCA results with larger fonts and highlight nearest neighbors."""
    plt.figure(figsize=(10, 8))

    # Assign colors
    colors = []
    for variant in pca_df.index:
        if variant == query_variant:
            colors.append('red')  # Query variant
        elif variant in nearest_variants:
            colors.append('green')  # Nearest neighbors
        else:
            colors.append('blue')  # Other variants

    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=colors)

    # Increase font sizes
    plt.rcParams.update({'font.size': 14})

    for variant_name in pca_df.index:
        plt.annotate(variant_name, (pca_df.loc[variant_name, 'PC1'], pca_df.loc[variant_name, 'PC2']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

    plt.title('PCA of Variant Mutation Effects with Nearest Neighbors', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.grid(True)
    plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate cumulative detrimental scores for protein variants using a WT proxy.")
    parser.add_argument("wt_proxy_fasta", help="FASTA file containing the WT proxy sequence.")
    parser.add_argument("variants_fasta", help="FASTA file containing variant sequences (including WT).")
    parser.add_argument("proxy_mutation_data", help="CSV file containing WT proxy mutation data with delta_log_prob scores.")
    parser.add_argument("output_csv", help="Output CSV file to save cumulative scores.")
    parser.add_argument("--insertion_score", type=float, default=-1.0,
                        help="Detrimental score to assign to insertions.")
    parser.add_argument("--deletion_score", type=float, default=-1.0,
                        help="Detrimental score to assign to deletions.")
    parser.add_argument("--perform_pca", action="store_true",
                        help="Perform PCA on mutation effect data.")
    parser.add_argument("--query_variant", type=str, default=None,
                        help="Variant name to use as query for nearest neighbor search.")
    parser.add_argument("--num_neighbors", type=int, default=5,
                        help="Number of nearest neighbors to find using Annoy.")
    args = parser.parse_args()

    # Load sequences
    proxy_sequences = load_sequences(args.wt_proxy_fasta)
    if len(proxy_sequences) != 1:
        raise ValueError("WT proxy FASTA file should contain exactly one sequence.")
    proxy_sequence = next(iter(proxy_sequences.values()))
    proxy_name = next(iter(proxy_sequences.keys()))

    variants = load_sequences(args.variants_fasta)

    # Load WT proxy mutation data
    proxy_mutation_data = pd.read_csv(args.proxy_mutation_data)
    # Remove any prefix from column names if necessary
    proxy_mutation_data.columns = [col.split('_')[-1] for col in proxy_mutation_data.columns]

    # Calculate cumulative scores
    variant_cumulative_scores, mutations_dict, variant_scores = calculate_cumulative_scores(
        proxy_sequence, variants, proxy_mutation_data,
        insertion_score=args.insertion_score,
        deletion_score=args.deletion_score
    )

    # Save cumulative scores to CSV
    scores_df = pd.DataFrame.from_dict(variant_cumulative_scores, orient='index', columns=['Cumulative Score'])
    scores_df.to_csv(args.output_csv)
    print("\nCumulative scores saved to '{}'".format(args.output_csv))

    # Prepare data for PCA and Annoy
    # Compile all mutations across variants
    all_mutations = set()
    for mutations in mutations_dict.values():
        for mut in mutations:
            mutation_id = f"{mut['mutation_type']}_{mut.get('mutation', '')}_{mut['position']}"
            all_mutations.add(mutation_id)

    all_mutations = sorted(all_mutations)
    mutation_index = {mutation: idx for idx, mutation in enumerate(all_mutations)}

    # Construct feature vectors
    variant_vectors = {}
    for variant_name, mutations in mutations_dict.items():
        scores = assign_mutation_scores(mutations, proxy_mutation_data,
                                        insertion_score=args.insertion_score,
                                        deletion_score=args.deletion_score)
        vector = np.zeros(len(all_mutations))
        for mut, score in zip(mutations, scores):
            mutation_id = f"{mut['mutation_type']}_{mut.get('mutation', '')}_{mut['position']}"
            idx = mutation_index[mutation_id]
            vector[idx] = score
        variant_vectors[variant_name] = vector

    # Perform PCA if requested
    if args.perform_pca:
        # Perform PCA
        pca_df, pca, df_for_annoy = perform_pca(variant_vectors)
        print("\nPCA Results:")
        print(pca_df)

        # Build Annoy Index
        f = df_for_annoy.shape[1]  # Number of features
        t = AnnoyIndex(f, 'angular')  # 'angular' distance uses cosine similarity
        variant_indices = {}
        for idx, variant_name in enumerate(df_for_annoy.index):
            t.add_item(idx, df_for_annoy.loc[variant_name].values)
            variant_indices[variant_name] = idx
        t.build(10)  # Build the index with 10 trees

        # Determine the query variant
        if args.query_variant is None:
            query_variant = list(variants.keys())[0]  # Default to the first variant
            print(f"No query variant specified. Using '{query_variant}' as the query variant.")
        else:
            query_variant = args.query_variant
            if query_variant not in variant_indices:
                raise ValueError(f"Query variant '{query_variant}' not found in the variants.")

        # Find nearest neighbors
        query_idx = variant_indices[query_variant]
        nn_indices = t.get_nns_by_item(query_idx, args.num_neighbors + 1)  # +1 to include the query itself
        nn_indices.remove(query_idx)  # Remove the query itself from the list
        nearest_variants = [df_for_annoy.index[i] for i in nn_indices]
        print(f"Nearest neighbors to '{query_variant}': {nearest_variants}")

        # Plot PCA with highlighted nearest neighbors
        plot_pca_with_annoy(pca_df, query_variant, nearest_variants)

if __name__ == "__main__":
    main()
