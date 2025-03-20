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