# 🏡 Boston Housing Price Prediction Pipeline

A production-grade machine learning project implementing custom scikit-learn pipelines to predict residential housing prices.

## 🎯 Project Overview
The objective of this project is to build a robust, reproducible regression pipeline to predict house values (`MEDV`). The project focuses on data leakage prevention, modular pipeline design, feature selection based on tree importance, and comparative analysis between baseline Decision Trees and Random Forests.

## 🔑 Project Findings & Methodology
1. **Bias & Redundancy Reduction:** Dropped demographic column `B` to mitigate algorithmic bias and dropped `CHAS` based on low predictive feature importance ($\approx 0.17\%$).
2. **Dominant Drivers:** Feature importance analysis revealed that `RM` (average rooms) and `LSTAT` (% lower status population) drive **~81.8%** of model predictions.
3. **Outlier Resilience:** Utilized tree ensemble models (**RandomForestRegressor**) to handle high-range outliers naturally without requiring monotonic skewness transformations on feature spaces.
4. **Pipeline Encapsulation:** Enforced custom `ColumnDropper` transformers directly inside `sklearn.pipeline.Pipeline` objects to allow direct prediction on raw, unseen test data.

## 📊 Performance Comparison
| Model | R² Score | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| **Decision Tree Regressor** | 0.8775 | 8.9798 | 2.2176 |
| **Random Forest Regressor** | **0.8900** | **8.0661** | **2.0410** |

> **Key Takeaway:** Random Forest achieved a higher $R^2$ score ($89.00\%$) and lower error metrics across all tests by averaging ensemble predictions to reduce variance.

## 🚀 How to Run
```bash
# Clone repository
git clone [https://github.com/your-username/boston-housing-pipeline.git](https://github.com/your-username/boston-housing-pipeline.git)
cd boston-housing-pipeline

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run training & evaluation script
python main.py
