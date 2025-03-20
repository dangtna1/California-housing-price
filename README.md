# Housing Price Prediction Project

This project focuses on building a machine learning model to predict median housing prices in California districts in America. It encompasses the entire ML pipeline from data acquisition to model evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Workflow](#workflow)
  - [Data Fetching & Exploration](#1-data-fetching--exploration)
  - [Stratified Train-Test Split](#2-stratified-train-test-split)
  - [Data Preprocessing](#3-data-preprocessing)
  - [Model Training & Evaluation](#4-model-training--evaluation)
  - [Hyperparameter Tuning](#5-hyperparameter-tuning)
  - [Final Evaluation](#6-final-evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Project Overview
**Objective**: Predict median house values using California census data  
**Key Features**:
- 10 numerical features (median income, housing median age, etc.)
- 1 categorical feature (ocean proximity)  
**Models Used**: Linear Regression, Decision Trees, Random Forest, SVM  
**Best Model**: Random Forest (RMSE: $48,209???)

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction
2. Install dependencies
   ```base
   pip install -r requirements.txt
## Dataset
The dataset is sourced from a publicly available repository:
* [Click here to download the dataset](https://raw.githubusercontent.com/dangtna1/datasets/refs/heads/main/housing.tgz)
* Description: Contains housing data for California districts, including features like longitude, latitude, median income, and ocean proximity.
* Size: 20,640 rows and 10 columns.
* Features:
    * longitude/latitude coordinates
    * housing_median_age
    * total_rooms
    * total_bedrooms
    * population
    * households
    * median_income
    * median_house_value
    * ocean_proximity (categorical)
## Workflow
### 1. Data Fetching & Exploration
**Automated Data Pipeline:**
   ```python
def fetch_housing_data():
    """Auto-download and extract dataset"""
    # Handles HTTP fetch and tar extraction
    
def load_housing_data():
    """Returns cleaned DataFrame"""
    return pd.read_csv(os.path.join(HOUSING_PATH, "housing.csv"))
```

**Key Insights:**
* 207 districts missing `total_bedrooms`
* `ocean_proximity` distribution
   * <1H OCEAN: 9136
   * INLAND: 6551
   * NEAR OCEAN: 2658
   * NEAR BAY: 2290
   * ISLAND: 5
* The distribution of each variable
![Histogram](images\distribution.png)
### 2. Stratified Train-Test Split
 ```python
# Create income strata
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0, 1.5, 3, 4.5, 6, np.inf],
                              labels=[1, 2, 3, 4, 5])

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

**Stratification Results:**
| income_cat | overall_proportion     | strat_proportion       | random_proportion      |
|------------|-----------------------|------------------------|------------------------|
| 1          | 0.039825581395348836  | 0.03997093023255814    | 0.04433139534883721    |
| 2          | 0.3188468992248062    | 0.3187984496124031     | 0.31443798449612403    |
| 3          | 0.3505813953488372    | 0.3505329457364341     | 0.3483527131782946     |
| 4          | 0.17630813953488372   | 0.17635658914728683    | 0.18023255813953487    |
| 5          | 0.11443798449612404   | 0.11434108527131782    | 0.11264534883720931    |

When we compare the distribution of income categories among data in *overall data*, *stratified sampled data*, and *random sampled data*. We can see that the distribution of income categories with *stratified* methods is closely the same as the distribution in *overall data*. This method can prevent bias when we sample which is likely to help build a good model
### 3. Data Preprocessing
**Pipeline Architecture:**
```
graph TD
    A[Raw Data] --> B[ColumnTransformer]
    B -->|Numerical| C[Impute Missing]
    C --> D[Add Features]
    D --> E[Scale]
    B -->|Categorical| F[OneHotEncode]
    E --> G[Combined Features]
    F --> G
    G --> H[Ready for Model]
```
### 4. Model Training & Evaluation

### 5. Hyperparameter Tuning

### 6. Final Evaluation

## Results

## Future Improvements
### 1. Feature Engineering:
* Incorporate real-time economic indicators
* Add walkability scores from external APIs
### 2. Model Enhancements:
* Experiment with Gradient Boosted Trees (XGBoost/LightGBM)
* Implement stacked ensemble models
### 3. Deployment and Monitor:
* Add data drift detection
* Implement MLflow for experiment tracking
