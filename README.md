# Customer Segmentation Model

This project performs customer segmentation using clustering algorithms on mall customer data.

## Dataset

- `data/Mall_Customers.csv`: Contains customer demographic and spending data which you can download from kaggle.

## Features Used

- Annual Income (k$)
- Spending Score (1-100)

## Clustering Methods

- **K-Means Clustering**: Used to segment customers into groups based on income and spending score. The optimal number of clusters is determined using the Elbow Method.
- **DBSCAN**: Density-based clustering to identify core samples and outliers.

## Usage

1. Install dependencies:
    ```sh
    pip install pandas scikit-learn matplotlib
    ```
2. Run the model:
    ```sh
    python src/model.py
    ```

## Output

- Displays cluster visualizations for both K-Means and DBSCAN.
- Prints summary statistics for each cluster.

## File Structure

```
data/
    Mall_Customers.csv
```
