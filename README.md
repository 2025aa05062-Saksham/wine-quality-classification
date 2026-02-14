# Wine Quality Classification - ML Assignment 2

## M.Tech AIML/DSE | BITS Pilani | Machine Learning

---

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a wine is of **good quality** (quality score >= 7) or **bad quality** (quality score < 7) based on its physicochemical properties. This is framed as a **binary classification** problem using 6 different ML models, with a focus on evaluating model performance across multiple metrics and deploying an interactive Streamlit web application for demonstration.

---

## Dataset Description

| Attribute | Details |
|-----------|---------|
| **Source** | [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| **Citation** | Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI ML Repository. |
| **Total Instances** | 6,497 (1,599 Red + 4,898 White) |
| **Features** | 12 (11 physicochemical + 1 wine type indicator) |
| **Target** | Binary (Good Wine = 1, Bad Wine = 0) |
| **Missing Values** | None |

### Feature Details

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Fixed acidity | Continuous | Tartaric acid concentration (g/dm3) |
| 2 | Volatile acidity | Continuous | Acetic acid concentration (g/dm3) |
| 3 | Citric acid | Continuous | Citric acid concentration (g/dm3) |
| 4 | Residual sugar | Continuous | Sugar remaining after fermentation (g/dm3) |
| 5 | Chlorides | Continuous | Sodium chloride concentration (g/dm3) |
| 6 | Free sulfur dioxide | Continuous | Free form of SO2 (mg/dm3) |
| 7 | Total sulfur dioxide | Continuous | Total SO2 (mg/dm3) |
| 8 | Density | Continuous | Density of wine (g/cm3) |
| 9 | pH | Continuous | Acidity level (0-14 scale) |
| 10 | Sulphates | Continuous | Potassium sulphate concentration (g/dm3) |
| 11 | Alcohol | Continuous | Alcohol by volume (%) |
| 12 | Wine type | Binary | 0 = Red wine, 1 = White wine |

### Target Variable
- **Good Wine (1)**: Original quality score >= 7 (~21.6% of dataset)
- **Bad Wine (0)**: Original quality score < 7 (~78.4% of dataset)

The dataset is **imbalanced**, with approximately 4:1 ratio of bad to good wines.

---

## Models Used

All 6 ML models were implemented on the same dataset with an 80/20 train-test split (stratified) and random_state=42.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|------|------|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8508 | 0.8019 | 0.6220 | 0.6172 | 0.6196 | 0.5268 |
| kNN | 0.8362 | 0.8307 | 0.6039 | 0.4883 | 0.5400 | 0.4453 |
| Naive Bayes | 0.7392 | 0.7494 | 0.3955 | 0.6133 | 0.4809 | 0.3310 |
| Random Forest (Ensemble) | 0.8869 | 0.9144 | 0.8011 | 0.5664 | 0.6636 | 0.6110 |
| XGBoost (Ensemble) | 0.8785 | 0.9004 | 0.7450 | 0.5820 | 0.6535 | 0.5877 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieves 82.23% accuracy as a baseline linear model. Strong AUC of 0.8048 indicates good probability calibration, but very low recall (0.2617) reveals it misses most good wines. The linear decision boundary cannot capture the complex non-linear interactions between physicochemical features. Precision is moderate (0.6147), meaning when it does predict good wine, it is reasonably correct. Low MCC (0.3178) confirms poor overall classification balance. |
| **Decision Tree** | Shows improved and balanced performance with 85.08% accuracy. The most notable strength is its balanced precision (0.6220) and recall (0.6172), resulting in the best F1 (0.6196) among non-ensemble models. With max_depth=10, it captures non-linear patterns while avoiding excessive overfitting. MCC of 0.5268 reflects strong classification quality. Decision rules are easily interpretable, making it suitable for understanding feature-level decision logic. |
| **kNN** | With k=7 and StandardScaler applied, achieves 83.62% accuracy. It has a competitive AUC of 0.8307 (third highest), indicating decent probability ranking. However, recall at 0.4883 means it misses about half the good wines. Performance is sensitive to feature scaling and the choice of k. Works well when decision boundaries are locally defined but struggles with global patterns in this 12-dimensional feature space. |
| **Naive Bayes** | Records the lowest accuracy (73.92%) due to the strong feature independence assumption, which does not hold for correlated physicochemical properties (e.g., pH and acidity, density and alcohol). However, it achieves the highest recall (0.6133) among non-ensemble models, making it useful when catching all good wines is the priority. Low precision (0.3955) means many false positives. Fastest model to train and predict. |
| **Random Forest (Ensemble)** | The best overall performer with the highest accuracy (88.69%), AUC (0.9144), precision (0.8011), F1 (0.6636), and MCC (0.6110). By aggregating 200 decision trees with max_depth=15, it significantly reduces overfitting while capturing complex feature interactions. The high precision means its positive predictions are highly reliable. Feature importance analysis reveals alcohol content, volatile acidity, and density as the most predictive features. Robust to outliers and noise in the dataset. |
| **XGBoost (Ensemble)** | Achieves the second-best results across nearly all metrics — accuracy 87.85%, AUC 0.9004, F1 0.6535, and MCC 0.5877. The gradient boosting approach sequentially corrects errors, achieving strong generalization with 200 estimators and learning_rate=0.1. Slightly lower precision than Random Forest (0.7450 vs 0.8011) but better recall (0.5820 vs 0.5664), suggesting a slightly different precision-recall trade-off. Built-in regularization (max_depth=6) prevents overfitting effectively. Both ensemble methods clearly outperform all individual models. |

### Key Takeaways

1. **Ensemble models (Random Forest and XGBoost) significantly outperform individual models** across all metrics, demonstrating the power of combining multiple learners.
2. **Class imbalance** (78.4% bad vs 21.6% good wines) heavily impacts recall. Logistic Regression in particular struggles with only 26.17% recall.
3. **Random Forest** achieves the best balance across all evaluation metrics with the highest MCC (0.6110), making it the recommended model for this dataset.
4. **Naive Bayes** offers the best recall among non-ensemble models if the priority is minimizing missed good wines, despite lower overall accuracy.
5. **Alcohol content, volatile acidity, and density** are consistently the most important features across tree-based models.
6. **AUC scores** show that even models with lower accuracy (like kNN at 0.8307) can rank predictions well, indicating potential improvement with threshold tuning.

---

## Project Structure

```
wine-quality-classification/
|-- app.py                    # Streamlit web application
|-- requirements.txt          # Python dependencies
|-- README.md                 # This file
|-- model/
    |-- train_models.ipynb    # Jupyter notebook with all 6 model implementations
```

---

## How to Run Locally

1. Clone the repository:
```bash
git clone <your-repo-url>
cd wine-quality-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

---

## Streamlit App Features

- **CSV Upload**: Upload your own test data for evaluation
- **Model Selection**: Dropdown to choose from 6 classification models
- **Metrics Display**: Interactive comparison table with all 6 evaluation metrics
- **Confusion Matrix**: Visual confusion matrix for each selected model
- **Classification Report**: Detailed precision, recall, F1 per class
- **ROC Curves**: Overlay ROC curves for all models
- **Dataset Explorer**: Browse data and feature distributions

---

## Deployment

Deployed on **Streamlit Community Cloud**: [INSERT YOUR STREAMLIT APP LINK HERE]

---

## Technologies Used

- Python 3.9+
- Scikit-learn
- XGBoost
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
