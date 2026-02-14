"""
Wine Quality Classification - Streamlit Web Application
=========================================================
This app provides an interactive UI for demonstrating 6 ML classification
models trained on the Wine Quality dataset (UCI ML Repository).

Features:
  - Dataset upload option (CSV)
  - Model selection dropdown
  - Display of evaluation metrics
  - Confusion matrix and classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #722F37;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_default_data():
    """Load and prepare the Wine Quality dataset."""
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    try:
        red = pd.read_csv(red_url, sep=';')
        white = pd.read_csv(white_url, sep=';')
    except Exception:
        red = pd.read_csv("winequality-red.csv", sep=';')
        white = pd.read_csv("winequality-white.csv", sep=';')
    
    red['wine_type'] = 0
    white['wine_type'] = 1
    df = pd.concat([red, white], ignore_index=True)
    
    # Binary classification: Good (quality >= 7) vs Bad (quality < 7)
    df['target'] = (df['quality'] >= 7).astype(int)
    
    return df


def get_feature_columns():
    """Return the list of feature columns used for modeling."""
    return [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'wine_type'
    ]


def train_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and return results."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled features for LR and kNN
        if name in ["Logistic Regression", "kNN"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_prob), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        trained_models[name] = model
    
    return results, trained_models, scaler


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix for a given model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='RdPu',
        xticklabels=['Bad Wine (0)', 'Good Wine (1)'],
        yticklabels=['Bad Wine (0)', 'Good Wine (1)'],
        ax=ax, linewidths=0.5, linecolor='white'
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_roc_curves(y_test, results):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.4f})", color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results):
    """Bar chart comparing all models across metrics."""
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    models_list = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        values = [results[m][metric] for m in models_list]
        short_names = ['LR', 'DT', 'kNN', 'NB', 'RF', 'XGB']
        bars = ax.bar(short_names, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================
# MAIN APPLICATION
# ============================================================

# Header
st.markdown('<p class="main-header">🍷 Wine Quality Classification</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Model ML Classification | UCI Wine Quality Dataset | '
    'Binary Classification: Good (quality ≥ 7) vs Bad (quality < 7)</p>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Dataset upload option
    st.markdown("### 📁 Upload Test Data (CSV)")
    uploaded_file = st.file_uploader(
        "Upload your own CSV test data",
        type=["csv"],
        help="Upload a CSV file with the same features as the Wine Quality dataset. "
             "Must include columns: fixed acidity, volatile acidity, citric acid, "
             "residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, "
             "density, pH, sulphates, alcohol, wine_type, and optionally 'quality' or 'target'."
    )
    
    st.markdown("---")
    
    # Model selection dropdown
    st.markdown("### 🤖 Select Model")
    selected_model = st.selectbox(
        "Choose a classification model",
        options=[
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ],
        index=4  # Default to Random Forest
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    - **Source**: UCI ML Repository
    - **Instances**: 6,497 (Red + White)
    - **Features**: 12
    - **Task**: Binary Classification
    - **Classes**: Good Wine (≥7) / Bad Wine (<7)
    """)
    
    st.markdown("---")
    st.markdown(
        "**Built for**: ML Assignment 2  \n"
        "**Program**: M.Tech AIML/DSE  \n"
        "**Institution**: BITS Pilani"
    )

# ============================================================
# LOAD DATA AND TRAIN MODELS
# ============================================================
with st.spinner("Loading data and training models... This may take a moment."):
    df = load_default_data()
    feature_cols = get_feature_columns()
    
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle uploaded test data
    using_uploaded = False
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Check if all feature columns exist
            missing_cols = [c for c in feature_cols if c not in uploaded_df.columns]
            if missing_cols:
                st.sidebar.error(f"Missing columns: {missing_cols}")
            else:
                X_test_upload = uploaded_df[feature_cols]
                
                if 'target' in uploaded_df.columns:
                    y_test_upload = uploaded_df['target']
                elif 'quality' in uploaded_df.columns:
                    y_test_upload = (uploaded_df['quality'] >= 7).astype(int)
                else:
                    y_test_upload = None
                
                if y_test_upload is not None:
                    X_test = X_test_upload
                    y_test = y_test_upload
                    using_uploaded = True
                    st.sidebar.success(f"✅ Loaded {len(X_test)} test samples from uploaded file!")
                else:
                    st.sidebar.warning("No target/quality column found. Using default test data for metrics.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    
    results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test)

# ============================================================
# MAIN CONTENT - TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "📊 Model Comparison", "🔍 Selected Model",
    "📈 Visualizations", "📄 Dataset Explorer"
])

# ------ TAB 1: OVERVIEW ------
with tab1:
    st.markdown("### 📋 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Instances", f"{len(df):,}")
    with col2:
        st.metric("Features", f"{len(feature_cols)}")
    with col3:
        st.metric("Train Samples", f"{len(X_train):,}")
    with col4:
        st.metric("Test Samples", f"{len(X_test):,}")
    
    if using_uploaded:
        st.info("📁 Currently using **uploaded test data** for evaluation.")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Problem Statement")
    st.markdown(
        "Predict whether a wine is of **good quality** (quality score ≥ 7) or "
        "**bad quality** (quality score < 7) based on 12 physicochemical properties. "
        "The dataset combines red and white Portuguese \"Vinho Verde\" wines from the "
        "UCI Machine Learning Repository."
    )
    
    st.markdown("### 📂 Dataset Description")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Source**: [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
        
        **Features (12)**:
        1. Fixed acidity
        2. Volatile acidity
        3. Citric acid
        4. Residual sugar
        5. Chlorides
        6. Free sulfur dioxide
        7. Total sulfur dioxide
        8. Density
        9. pH
        10. Sulphates
        11. Alcohol
        12. Wine type (0=Red, 1=White)
        """)
    with col2:
        st.markdown("""
        **Target**: Binary classification
        - **Good Wine (1)**: Quality score ≥ 7
        - **Bad Wine (0)**: Quality score < 7
        
        **Class Distribution**:
        """)
        class_dist = df['target'].value_counts()
        st.dataframe(pd.DataFrame({
            'Class': ['Bad Wine (0)', 'Good Wine (1)'],
            'Count': [class_dist[0], class_dist[1]],
            'Percentage': [f"{class_dist[0]/len(df)*100:.1f}%", f"{class_dist[1]/len(df)*100:.1f}%"]
        }), hide_index=True)

# ------ TAB 2: MODEL COMPARISON ------
with tab2:
    st.markdown("### 📊 Model Comparison Table")
    
    # Build comparison dataframe
    comparison_data = []
    for name in results:
        row = {'ML Model Name': name}
        row.update({k: v for k, v in results[name].items() if k not in ['y_pred', 'y_prob']})
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Style the dataframe
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_max]
    
    styled_df = comparison_df.style.apply(
        highlight_max,
        subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    ).format({
        'Accuracy': '{:.4f}', 'AUC': '{:.4f}', 'Precision': '{:.4f}',
        'Recall': '{:.4f}', 'F1': '{:.4f}', 'MCC': '{:.4f}'
    })
    
    st.dataframe(styled_df, hide_index=True, use_container_width=True)
    
    st.markdown("*Green highlighted cells indicate the best score for each metric.*")
    
    # Best model summary
    st.markdown("---")
    st.markdown("### 🏆 Best Model by Metric")
    
    best_cols = st.columns(6)
    metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    for col, metric in zip(best_cols, metric_names):
        best_model = max(results, key=lambda m: results[m][metric])
        with col:
            st.markdown(f"**{metric}**")
            st.markdown(f"🥇 {best_model}")
            st.markdown(f"`{results[best_model][metric]:.4f}`")
    
    # Observations table
    st.markdown("---")
    st.markdown("### 📝 Model Performance Observations")
    
    observations = {
        "Logistic Regression": (
            "Serves as a solid baseline with reasonable accuracy. Being a linear model, it may struggle "
            "with non-linear decision boundaries in the feature space. It performs well on AUC, "
            "indicating good probability calibration, but recall for the minority class (good wine) "
            "may be limited due to class imbalance."
        ),
        "Decision Tree": (
            "Captures non-linear patterns and feature interactions naturally. Prone to overfitting "
            "on training data if not properly pruned (max_depth=10 used). Provides easily interpretable "
            "rules but may have lower generalization compared to ensemble methods. Precision and recall "
            "balance depends heavily on the tree structure."
        ),
        "kNN": (
            "Performance is sensitive to the choice of k and feature scaling (StandardScaler applied). "
            "Works well when decision boundaries are locally defined. Can be computationally expensive "
            "at prediction time for large datasets. May struggle with high-dimensional feature spaces "
            "due to the curse of dimensionality."
        ),
        "Naive Bayes": (
            "Fast and efficient, works well with the Gaussian assumption for continuous features. "
            "The strong independence assumption between features may not hold for physicochemical "
            "properties (e.g., pH and acidity are correlated), potentially limiting performance. "
            "Typically achieves good recall but may sacrifice precision."
        ),
        "Random Forest (Ensemble)": (
            "Combines multiple decision trees to reduce overfitting and improve generalization. "
            "Generally one of the top performers with high accuracy, AUC, and balanced F1 scores. "
            "Provides feature importance rankings, revealing that alcohol content and volatile acidity "
            "are among the most predictive features. Robust to outliers and noise."
        ),
        "XGBoost (Ensemble)": (
            "Advanced gradient boosting method that sequentially builds trees to correct errors. "
            "Typically achieves the best or near-best performance across all metrics. Handles class "
            "imbalance well with its gradient-based optimization. The learning rate and tree depth "
            "hyperparameters allow fine-tuning for optimal performance. Often the best MCC score, "
            "indicating strong overall classification quality."
        )
    }
    
    obs_data = [{"ML Model Name": k, "Observation": v} for k, v in observations.items()]
    st.dataframe(pd.DataFrame(obs_data), hide_index=True, use_container_width=True)

# ------ TAB 3: SELECTED MODEL DETAILS ------
with tab3:
    st.markdown(f"### 🔍 Detailed Results: {selected_model}")
    
    res = results[selected_model]
    
    # Metric cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    metrics_display = [
        (col1, "Accuracy", res['Accuracy']),
        (col2, "AUC", res['AUC']),
        (col3, "Precision", res['Precision']),
        (col4, "Recall", res['Recall']),
        (col5, "F1 Score", res['F1']),
        (col6, "MCC", res['MCC'])
    ]
    
    for col, label, val in metrics_display:
        with col:
            st.metric(label, f"{val:.4f}")
    
    st.markdown("---")
    
    # Confusion matrix and classification report side by side
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, res['y_pred'], selected_model)
        st.pyplot(fig_cm)
    
    with col_right:
        st.markdown("#### Classification Report")
        report = classification_report(
            y_test, res['y_pred'],
            target_names=['Bad Wine (0)', 'Good Wine (1)'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
        
        # ROC curve for selected model
        st.markdown("#### ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax_roc.plot(fpr, tpr, color='#722F37', linewidth=2, label=f'AUC = {res["AUC"]:.4f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve - {selected_model}')
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)

# ------ TAB 4: VISUALIZATIONS ------
with tab4:
    st.markdown("### 📈 Visual Comparisons")
    
    # ROC Curves - All Models
    st.markdown("#### ROC Curves - All Models")
    fig_all_roc = plot_roc_curves(y_test, results)
    st.pyplot(fig_all_roc)
    
    st.markdown("---")
    
    # Metrics Comparison Bar Charts
    st.markdown("#### Performance Metrics Comparison")
    fig_bars = plot_metrics_comparison(results)
    st.pyplot(fig_bars)

# ------ TAB 5: DATASET EXPLORER ------
with tab5:
    st.markdown("### 📄 Dataset Explorer")
    
    st.markdown("#### Sample Data (First 20 rows)")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Feature Statistics")
    st.dataframe(df[feature_cols].describe().T.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", feature_cols)
    
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    df[df['target'] == 0][selected_feature].hist(
        ax=axes_hist[0], bins=30, alpha=0.7, color='#e74c3c', label='Bad Wine'
    )
    df[df['target'] == 1][selected_feature].hist(
        ax=axes_hist[0], bins=30, alpha=0.7, color='#2ecc71', label='Good Wine'
    )
    axes_hist[0].set_title(f'Distribution of {selected_feature}', fontweight='bold')
    axes_hist[0].legend()
    axes_hist[0].grid(True, alpha=0.3)
    
    # Box plot
    df.boxplot(column=selected_feature, by='target', ax=axes_hist[1])
    axes_hist[1].set_title(f'{selected_feature} by Wine Quality', fontweight='bold')
    axes_hist[1].set_xlabel('Target (0=Bad, 1=Good)')
    plt.suptitle('')  # Remove auto title
    plt.tight_layout()
    st.pyplot(fig_hist)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85rem;'>"
    "🍷 Wine Quality Classification App | ML Assignment 2 | M.Tech AIML/DSE | BITS Pilani"
    "</div>",
    unsafe_allow_html=True
)
