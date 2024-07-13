# Heart Failure Exploratory Data Analysis (EDA) Project

## Introduction

This project involves an exploratory data analysis (EDA) of a heart failure dataset. The goal is to analyze various factors that contribute to heart failure and visualize the data to gain insights. This analysis includes handling missing values, identifying and removing duplicates, and visualizing relationships between features and the target variable (DEATH_EVENT).

## Dataset

The dataset used for this project is the Heart Failure Prediction dataset, which contains 299 observations with 13 features. The features include both numerical and categorical variables that provide information about the patients, such as age, gender, blood pressure, and various biomarkers.

## Project Structure

- **data/**: Contains the dataset used for the analysis.
- **notebooks/**: Contains the Jupyter notebooks used for the analysis.
- **images/**: Contains images of the visualizations created during the analysis.
- **scripts/**: Contains the Python scripts used for data analysis and visualization.
- **README.md**: This file.

## Dependencies

The following Python libraries are required to run the scripts and notebooks:

- numpy
- pandas
- matplotlib
- seaborn

You can install the required libraries using:

```sh
pip install numpy pandas matplotlib seaborn
```

## Data Preparation

1. **Importing Libraries**: Essential libraries like `numpy`, `pandas`, `matplotlib`, and `seaborn` are imported.
2. **Loading the Dataset**: The dataset is loaded into a pandas DataFrame.
3. **Data Cleaning**: 
   - Changing the data types of `age` and `platelets` from float to integer.
   - Checking for missing values and duplicates.
4. **Data Transformation**: 
   - Dividing features into categorical and numerical based on unique values.
   - Scaling numerical features for better interpretability.

## Data Analysis

### Descriptive Statistics

- Statistical summary of the dataset using `describe()`.
- Count of missing values and duplicates.

### Visualizations

1. **Heatmaps**: Visual representation of the mean values for patients with and without the DEATH_EVENT.
2. **Distribution Plots**: 
   - Categorical Features: Distribution of categorical features.
   - Numerical Features: Distribution of numerical features.
3. **Bar and Pie Charts**: Visualization of the target variable (DEATH_EVENT) counts and percentages.
4. **Box Plots**: Relationship between categorical features and numerical features with respect to the target variable.
5. **Scatter Plots**: Relationship between numerical features with respect to the target variable.

### Key Insights

- Patients without anaemia, diabetes, high blood pressure, and smoking habits have higher death rates than those with these conditions.
- Males are more prone to heart failure-related death events than females.
- Numerical features like age, creatinine phosphokinase, ejection fraction, platelets, serum creatinine, and serum sodium show distinct patterns associated with the DEATH_EVENT.

## Summary

This project provides a comprehensive exploratory data analysis of the heart failure dataset. The analysis helps in understanding the key factors contributing to heart failure and provides a basis for further predictive modeling.

## Repository Contents

- `data/heart.csv`: The dataset used for analysis.
- `notebooks/EDA_Heart_Failure.ipynb`: Jupyter notebook containing the EDA.
- `images/`: Directory containing images of visualizations.
- `scripts/eda.py`: Python script for running the EDA.
- `README.md`: Project documentation.

## Conclusion

This analysis reveals significant insights into the factors contributing to heart failure. The visualizations and statistical summaries provide a clear understanding of the dataset and highlight the importance of specific features in predicting heart failure outcomes.
