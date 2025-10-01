#!/usr/bin/eny python3
# -*- coding: utf-8 -*-





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def print_and_plot_item_parameters(itembank_csv_file: str):
    """
    Load item parameters from a CSV file and generate plots and summary
    """
    # Read your CSV file
    df = pd.read_csv(itembank_csv_file)  # Replace with your actual file path

    # Set up the plotting style
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribution of IRT Parameters', fontsize=16, fontweight='bold')

    # Plot 1: Discrimination (a) parameter
    axes[0, 0].hist(df['a'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['a'].mean(), color='red', linestyle='--', label=f'Mean: {df["a"].mean():.2f}')
    axes[0, 0].set_xlabel('Discrimination (a)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Discrimination Parameters')
    axes[0, 0].legend()
    axes[0, 0].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["a"].min():.2f}\nMax: {df["a"].max():.2f}\nStd: {df["a"].std():.2f}',
                    transform=axes[0, 0].transAxes, verticalalignment='top')

    # Plot 2: Difficulty (b) parameter
    axes[0, 1].hist(df['b'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(df['b'].mean(), color='red', linestyle='--', label=f'Mean: {df["b"].mean():.2f}')
    axes[0, 1].set_xlabel('Difficulty (b)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Difficulty Parameters')
    axes[0, 1].legend()
    axes[0, 1].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["b"].min():.2f}\nMax: {df["b"].max():.2f}\nStd: {df["b"].std():.2f}',
                    transform=axes[0, 1].transAxes, verticalalignment='top')

    # Plot 3: Guessing (c) parameter
    axes[1, 0].hist(df['c'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(df['c'].mean(), color='red', linestyle='--', label=f'Mean: {df["c"].mean():.6f}')
    axes[1, 0].set_xlabel('Guessing (c)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Guessing Parameters')
    axes[1, 0].legend()
    axes[1, 0].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["c"].min():.6f}\nMax: {df["c"].max():.6f}\nStd: {df["c"].std():.6f}',
                    transform=axes[1, 0].transAxes, verticalalignment='top')

    # Plot 4: Box plot of all parameters (standardized for comparison)
    parameter_data = []
    for idx, row in df.iterrows():
        parameter_data.extend([
            {'Parameter': 'a', 'Value': row['a'], 'Item': row['ids']},
            {'Parameter': 'b', 'Value': row['b'], 'Item': row['ids']},
            {'Parameter': 'c', 'Value': row['c'], 'Item': row['ids']}
        ])

    param_df = pd.DataFrame(parameter_data)
    sns.boxplot(data=param_df, x='Parameter', y='Value', ax=axes[1, 1], palette=['skyblue', 'lightgreen', 'orange'])
    axes[1, 1].set_title('Box Plot of All Parameters')

    # Remove the empty subplot if you have an odd number
    # fig.delaxes(axes[1, 1])  # Uncomment if you want only 3 plots

    plt.tight_layout()
    #plt.show() # interactive only

    plt.savefig('irt_parameters_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('irt_parameters_distribution.pdf', bbox_inches='tight')
    print("Plots saved as 'irt_parameters_distribution.png' and 'irt_parameters_distribution.pdf'")

    # Print summary statistics
    print("=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"Discrimination (a) parameter:")
    print(f"  Mean: {df['a'].mean():.4f}")
    print(f"  Std:  {df['a'].std():.4f}")
    print(f"  Min:  {df['a'].min():.4f}")
    print(f"  Max:  {df['a'].max():.4f}")
    print(f"  # Negative: {(df['a'] < 0).sum()} items")
    print()

    print(f"Difficulty (b) parameter:")
    print(f"  Mean: {df['b'].mean():.4f}")
    print(f"  Std:  {df['b'].std():.4f}")
    print(f"  Min:  {df['b'].min():.4f}")
    print(f"  Max:  {df['b'].max():.4f}")
    print()

    print(f"Guessing (c) parameter:")
    print(f"  Mean: {df['c'].mean():.6f}")
    print(f"  Std:  {df['c'].std():.6f}")
    print(f"  Min:  {df['c'].min():.6f}")
    print(f"  Max:  {df['c'].max():.6f}")
    print(f"  # Near zero (< 0.001): {(df['c'] < 0.001).sum()} items")
    print(f"  # Moderate (> 0.2): {(df['c'] > 0.2).sum()} items")

    # Additional analysis by item type if you have the 'correct' column
    if 'correct' in df.columns:
        print("\n" + "=" * 50)
        print("ANALYSIS BY ITEM TYPE")
        print("=" * 50)
        for item_type in df['correct'].unique():
            subset = df[df['correct'] == item_type]
            print(f"\n{item_type} items (N={len(subset)}):")
            print(f"  a: mean={subset['a'].mean():.4f}, range=[{subset['a'].min():.4f}, {subset['a'].max():.4f}]")
            print(f"  b: mean={subset['b'].mean():.4f}, range=[{subset['b'].min():.4f}, {subset['b'].max():.4f}]")
            print(f"  c: mean={subset['c'].mean():.6f}, range=[{subset['c'].min():.6f}, {subset['c'].max():.6f}]")

if __name__ == "__main__":
    import os
    current_source_dir = os.path.dirname(os.path.abspath(__file__))
    itembank_file = os.path.join(current_source_dir, 'adaptivetesting', 'tests', 'itembank_essential.csv')
    if not os.path.isfile(itembank_file):
        raise FileNotFoundError(f"Item bank file not found: {itembank_file}")
    print_and_plot_item_parameters(itembank_file)
