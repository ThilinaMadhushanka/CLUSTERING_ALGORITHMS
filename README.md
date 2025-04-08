# Clustering Analysis: A Comparative Study of Different Clustering Algorithms

## Overview
This project implements and compares various clustering algorithms on synthetic datasets. It provides a comprehensive analysis of different clustering techniques, their performance metrics, and visualizations of the results.

## Features
- Implementation of multiple clustering algorithms:
  - K-Means
  - Mini-Batch K-Means
  - DBSCAN
  - Agglomerative Clustering
  - Mean Shift
  - Spectral Clustering
  - Affinity Propagation
  - BIRCH
  - OPTICS
  - Gaussian Mixture Models

- Comprehensive evaluation metrics:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index
  - Execution Time

- Visualization capabilities:
  - Original dataset plots
  - Clustering results visualization
  - Performance metrics comparison
  - Best algorithm analysis

## Project Structure
```
.
├── Main.py                 # Main implementation file
├── report.md              # Detailed analysis report
├── convert_to_word.py     # Report conversion script
└── clustering_outputs/    # Directory for output files
    ├── all_datasets_grid.png
    ├── clustering_results_summary.csv
    ├── metrics_comparison_*.png
    └── best_algorithm_comparison.png
```

## Requirements
- Python 3.8+
- Required packages:
  ```
  numpy
  matplotlib
  scikit-learn
  pandas
  seaborn
  python-docx
  markdown
  beautifulsoup4
  Pillow
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main analysis:
   ```bash
   python Main.py
   ```

2. Generate the Word report:
   ```bash
   python convert_to_word.py
   ```

## Output Files
- `clustering_outputs/all_datasets_grid.png`: Visualization of all synthetic datasets
- `clustering_outputs/clustering_results_summary.csv`: Detailed results of all algorithms
- `clustering_outputs/metrics_comparison_*.png`: Performance metrics comparison plots
- `clustering_outputs/best_algorithm_comparison.png`: Overall algorithm comparison
- `Clustering_Analysis_Report.docx`: Comprehensive analysis report

## Analysis Methodology
1. **Dataset Generation**:
   - Anisotropic dataset
   - Circles dataset
   - Combined dataset

2. **Algorithm Evaluation**:
   - Performance metrics calculation
   - Execution time measurement
   - Cluster quality assessment

3. **Visualization and Reporting**:
   - Dataset visualization
   - Clustering results plotting
   - Performance metrics comparison
   - Best algorithm identification

## Results
The analysis compares the performance of different clustering algorithms across various metrics:
- Silhouette Score (higher is better)
- Calinski-Harabasz Index (higher is better)
- Davies-Bouldin Index (lower is better)
- Execution Time (lower is better)

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Scikit-learn library for algorithm implementations
- Matplotlib and Seaborn for visualization
- Python-docx for report generation 