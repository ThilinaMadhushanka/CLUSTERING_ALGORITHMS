#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clustering Analysis Assignment - Comparing Different Clustering Algorithms

This script demonstrates various clustering algorithms on synthetic datasets
and evaluates their performance using multiple metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering,
    MeanShift, SpectralClustering, AffinityPropagation, Birch, OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import os
import csv
import pandas as pd
import seaborn as sns

def generate_datasets(n_samples=1500, random_state=42):
    """Generate synthetic datasets."""
    # Generate anisotropic dataset using make_blobs with transformation
    X_aniso, y_aniso = make_blobs(n_samples=n_samples//2, centers=3, random_state=random_state)
    # Apply transformation to make it anisotropic
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X_aniso, transformation)
    
    # Generate circles dataset
    X_circles, y_circles = make_circles(n_samples=n_samples//2, factor=0.5, noise=0.05, random_state=random_state)
    
    # Create combined dataset
    X_combined = np.vstack([X_aniso, X_circles])
    y_combined = np.hstack([y_aniso, y_circles + 2])  # Add 2 to circles labels to distinguish from aniso
    
    return {
        'aniso': (X_aniso, y_aniso),
        'circles': (X_circles, y_circles),
        'combined_aniso_circles': (X_combined, y_combined)
    }

def get_clustering_algorithms():
    """Define clustering algorithms to be tested."""
    return {
        'KMeans': KMeans(n_clusters=3, random_state=42),
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=3, random_state=42),
        'AffinityPropagation': AffinityPropagation(random_state=42),
        'MeanShift': MeanShift(bandwidth=2),
        'SpectralClustering': SpectralClustering(n_clusters=3, random_state=42),
        'Ward': AgglomerativeClustering(n_clusters=3, linkage='ward'),
        'AgglomerativeClustering': AgglomerativeClustering(n_clusters=3),
        'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
        'OPTICS': OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1),
        'Birch': Birch(n_clusters=3),
        'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
    }

def evaluate_clustering(X, labels):
    """Evaluate clustering results using multiple metrics."""
    if len(np.unique(labels)) < 2:
        return None, None, None
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    return silhouette, calinski, davies

def plot_dataset(X, y, title, ax):
    """Plot a single dataset."""
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y if y is not None else 'blue',
                         s=50, alpha=0.7, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle='--', alpha=0.7)
    return scatter

def save_all_datasets_grid(datasets, output_dir):
    """Save a grid image showing all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        ax.scatter(X[:, 0], X[:, 1], c=y if y is not None else 'blue', s=40, alpha=0.7, cmap='viridis')
        ax.set_title(name.capitalize(), fontsize=14)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_datasets_grid.png"))
    plt.close()

def plot_clustering_results(dataset_name, X, all_labels, output_dir):
    """Plot all clustering results for a single dataset in one figure."""
    n_algorithms = len(all_labels)
    n_cols = 3  # Number of columns in the grid
    n_rows = (n_algorithms + n_cols) // n_cols  # Calculate number of rows needed
    
    # Create figure with adjusted size
    plt.figure(figsize=(20, 6 * n_rows))
    
    # Plot original dataset
    plt.subplot(n_rows, n_cols, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.7)
    plt.title(f"Original {dataset_name.capitalize()} Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot results for each algorithm
    for idx, (algo_name, result) in enumerate(all_labels.items(), start=2):
        if idx <= n_rows * n_cols:  # Ensure we don't exceed the grid size
            plt.subplot(n_rows, n_cols, idx)
            scatter = plt.scatter(X[:, 0], X[:, 1], 
                                c=result['labels'], 
                                cmap='viridis', 
                                s=50, 
                                alpha=0.7)
            plt.colorbar(scatter)
            plt.title(f"{algo_name}\nClusters: {result['n_clusters']}\n"
                     f"Silhouette: {result['silhouette']:.2f}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_all_algorithms.png"))
    plt.close()

def save_results_to_csv(all_results, output_dir):
    """Save all clustering results to a CSV file."""
    csv_path = os.path.join(output_dir, "clustering_results_summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Algorithm', 'Time (s)', 'Clusters', 'Silhouette', 'Calinski', 'Davies']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for dataset_name, dataset_results in all_results.items():
            for algo_name, result in dataset_results.items():
                writer.writerow({
                    'Dataset': dataset_name,
                    'Algorithm': algo_name,
                    'Time (s)': f"{result['time']:.4f}",
                    'Clusters': result['n_clusters'],
                    'Silhouette': f"{result['silhouette']:.4f}" if result['silhouette'] is not None else "N/A",
                    'Calinski': f"{result['calinski']:.2f}" if result['calinski'] is not None else "N/A",
                    'Davies': f"{result['davies']:.2f}" if result['davies'] is not None else "N/A"
                })
    print(f"\nResults saved to {csv_path}")

def plot_metrics_comparison(data, dataset_name, output_dir):
    dataset_df = data[data['Dataset'] == dataset_name]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    silhouette_palette = sns.color_palette("bright", n_colors=len(dataset_df))
    calinski_palette = sns.color_palette("deep", n_colors=len(dataset_df))
    davies_palette = sns.color_palette("Set2", n_colors=len(dataset_df))
    
    num_algorithms = len(dataset_df)
    bar_width = 0.6 / num_algorithms
    
    sns.barplot(x='Algorithm', y='Silhouette', data=dataset_df, ax=ax1, palette=silhouette_palette, width=0.6)
    ax1.set_title(f'Silhouette Score Comparison - {dataset_name} (Higher is better)', pad=20, fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    sns.barplot(x='Algorithm', y='Calinski', data=dataset_df, ax=ax2, palette=calinski_palette, width=0.6)
    ax2.set_title(f'Calinski-Harabasz Score Comparison - {dataset_name} (Higher is better)', pad=20, fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    sns.barplot(x='Algorithm', y='Davies', data=dataset_df, ax=ax3, palette=davies_palette, width=0.6)
    ax3.set_title(f'Davies-Bouldin Score Comparison - {dataset_name} (Lower is better)', pad=20, fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    for ax in [ax1, ax2, ax3]:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                       (p.get_x() + p.get_width()/2., p.get_height()),
                       ha='center', va='bottom', xytext=(0, 5),
                       textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_comparison_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_best_algorithm_comparison(data, output_dir):
    normalized_data = data.copy()
    
    for dataset in data['Dataset'].unique():
        dataset_mask = normalized_data['Dataset'] == dataset
        
        silhouette_max = normalized_data.loc[dataset_mask, 'Silhouette'].max()
        normalized_data.loc[dataset_mask, 'Silhouette'] = normalized_data.loc[dataset_mask, 'Silhouette'] / silhouette_max
        
        calinski_max = normalized_data.loc[dataset_mask, 'Calinski'].max()
        normalized_data.loc[dataset_mask, 'Calinski'] = normalized_data.loc[dataset_mask, 'Calinski'] / calinski_max
        
        davies_min = normalized_data.loc[dataset_mask, 'Davies'].min()
        normalized_data.loc[dataset_mask, 'Davies'] = davies_min / normalized_data.loc[dataset_mask, 'Davies']
    
    normalized_data['Combined_Score'] = (normalized_data['Silhouette'] + 
                                       normalized_data['Calinski'] + 
                                       normalized_data['Davies']) / 3
    
    plt.figure(figsize=(15, 8))
    
    num_algorithms = len(normalized_data['Algorithm'].unique())
    custom_palette = sns.color_palette("bright", n_colors=num_algorithms)
    
    bar_width = 0.6 / num_algorithms
    
    ax = sns.barplot(x='Dataset', y='Combined_Score', hue='Algorithm', 
                     data=normalized_data, palette=custom_palette,
                     saturation=0.9, width=0.6)
    
    num_datasets = len(normalized_data['Dataset'].unique())
    
    for i, bars in enumerate(ax.containers):
        for bar in bars:
            x = bar.get_x()
            bar.set_x(x + i * bar_width * 0.3)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='bottom', xytext=(0, 5), 
                   textcoords='offset points', fontsize=8)
    
    plt.title('Best Algorithm Comparison Across Datasets\n(Combined Score: Silhouette + Calinski-Harabasz + Davies-Bouldin)',
             pad=20, fontsize=12)
    plt.xlabel('Dataset', fontsize=10, labelpad=10)
    plt.ylabel('Normalized Combined Score', fontsize=10, labelpad=10)
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=8, title_fontsize=9)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'best_algorithm_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nBest Algorithm for Each Dataset:")
    print("-" * 50)
    for dataset in normalized_data['Dataset'].unique():
        best_algo = normalized_data[normalized_data['Dataset'] == dataset].nlargest(1, 'Combined_Score')
        print(f"{dataset}: {best_algo['Algorithm'].values[0]} (Score: {best_algo['Combined_Score'].values[0]:.3f})")

def main():
    output_dir = "clustering_outputs"
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # Generate datasets
    print("Generating datasets...")
    datasets = generate_datasets()

    # Save all datasets in a single grid image
    save_all_datasets_grid(datasets, output_dir)
    print("Saved all datasets grid image.")

    # Get clustering algorithms
    algorithms = get_clustering_algorithms()
    all_results = {}

    # Process each dataset
    for dataset_name, (X, y_true) in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dataset_results = {}

        for algo_name, algorithm in algorithms.items():
            print(f"Running {algo_name}...")
            start_time = time.time()
            if algo_name == 'MeanShift':
                algorithm.fit(X_scaled)
                labels = algorithm.labels_
            else:
                labels = algorithm.fit_predict(X_scaled)
            end_time = time.time()
            execution_time = end_time - start_time
            silhouette, calinski, davies = evaluate_clustering(X_scaled, labels)

            dataset_results[algo_name] = {
                'labels': labels,
                'time': execution_time,
                'silhouette': silhouette if silhouette is not None else 0,
                'calinski': calinski if calinski is not None else 0,
                'davies': davies if davies is not None else 0,
                'n_clusters': len(np.unique(labels))
            }

        # Create combined visualization for this dataset
        plot_clustering_results(dataset_name, X, dataset_results, output_dir)

        all_results[dataset_name] = dataset_results

        # Display evaluation metrics
        print(f"\nEvaluation Metrics for {dataset_name}:")
        print("-" * 80)
        print(f"{'Algorithm':<20} {'Time (s)':<10} {'Clusters':<10} {'Silhouette':<12} {'Calinski':<12} {'Davies':<10}")
        print("-" * 80)
        for algo_name, result in dataset_results.items():
            silhouette = f"{result['silhouette']:.4f}" if result['silhouette'] is not None else "N/A"
            calinski = f"{result['calinski']:.2f}" if result['calinski'] is not None else "N/A"
            davies = f"{result['davies']:.2f}" if result['davies'] is not None else "N/A"
            print(f"{algo_name:<20} {result['time']:.4f}     {result['n_clusters']:<10} {silhouette:<12} {calinski:<12} {davies:<10}")
        print("-" * 80)

    # Save results to CSV
    save_results_to_csv(all_results, output_dir)

    print("\nGenerating comparison plots...")
    
    # Read the CSV file with all results
    results_df = pd.read_csv(os.path.join(output_dir, 'clustering_results_summary.csv'))
    
    # Create separate comparison plots for each dataset
    for dataset_name in datasets.keys():
        print(f"Creating comparison plots for {dataset_name}...")
        plot_metrics_comparison(results_df, dataset_name, output_dir)
    
    # Create final best algorithm comparison
    print("\nCreating final best algorithm comparison...")
    plot_best_algorithm_comparison(results_df, output_dir)
    
    print("\nAll comparison plots have been generated and saved in the clustering_outputs directory.")

if __name__ == "__main__":
    main()