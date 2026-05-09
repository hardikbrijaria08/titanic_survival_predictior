"""
Titanic Survivor Prediction
============================
Uses XGBoost to predict passenger survival on the Titanic dataset.
Includes data preprocessing, feature engineering, model training,
evaluation, and feature importance analysis.

Author: Hardik Brijaria
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(filepath='titanic.csv'):
    """Load the Titanic dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        # If no file provided, use the built-in seaborn dataset
        print("📦 No CSV found — loading built-in seaborn Titanic dataset...")
        import seaborn as sns
        df = sns.load_dataset('titanic')
        # Rename to match Kaggle-style column names
        df.rename(columns={
            'survived': 'Survived',
            'pclass': 'Pclass',
            'sex': 'Sex',
            'age': 'Age',
            'sibsp': 'SibSp',
            'parch': 'Parch',
            'fare': 'Fare',
            'embarked': 'Embarked',
            'alone': 'Alone'
        }, inplace=True)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────

def preprocess(df):
    """Clean and engineer features from raw data."""
    df = df.copy()

    # Select relevant columns
    cols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # --- Handle Missing Values ---
    df['Age'].fillna(df['Age'].median(), inplace=True)         # fill missing age with median
    df['Fare'].fillna(df['Fare'].median(), inplace=True)       # fill missing fare with median
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # fill with most common port

    # --- Feature Engineering ---
    # Family size: total people travelling together
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Is the passenger alone?
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age group buckets
    df['AgeGroup'] = pd.cut(df['Age'],
                            bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

    # Fare tier
    df['FareTier'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Mid', 'High', 'VeryHigh'])

    # --- Encode Categorical Variables ---
    le = LabelEncoder()
    for col in ['Sex', 'Embarked', 'AgeGroup', 'FareTier']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    print(f"✅ Preprocessing done. Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df):
    """Split data and train XGBoost classifier."""
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    print("✅ Model trained successfully.")
    return model, X_train, X_test, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Print accuracy, classification report, and confusion matrix."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Did Not Survive', 'Survived'],
                yticklabels=['Did Not Survive', 'Survived'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("  📊 Confusion matrix saved as confusion_matrix.png")

    return acc


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names):
    """Plot and save feature importance chart."""
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title('Feature Importance — Titanic Survival Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("  📊 Feature importance saved as feature_importance.png")

    print("\n  Top Features Affecting Survival:")
    for _, row in feat_df.sort_values('Importance', ascending=False).head(5).iterrows():
        print(f"    • {row['Feature']}: {row['Importance']:.4f}")


# ─────────────────────────────────────────────
# 6. PREDICT A SINGLE PASSENGER
# ─────────────────────────────────────────────

def predict_passenger(model, feature_names):
    """Demo: predict survival for a custom passenger."""
    print(f"\n{'='*45}")
    print("  SAMPLE PREDICTION")
    print(f"{'='*45}")

    # Example passenger: 3rd class, male, 22 years old, alone
    sample = {
        'Pclass': 3, 'Sex': 1, 'Age': 22, 'SibSp': 0,
        'Parch': 0, 'Fare': 7.25, 'Embarked': 2,
        'FamilySize': 1, 'IsAlone': 1, 'AgeGroup': 2, 'FareTier': 0
    }
    sample_df = pd.DataFrame([sample])
    # Align columns with training data
    sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(sample_df)[0][1]
    pred = "Survived ✅" if prob >= 0.5 else "Did Not Survive ❌"
    print(f"  Passenger: 3rd class male, age 22, travelling alone")
    print(f"  Prediction: {pred}  (Survival probability: {prob:.1%})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚢 TITANIC SURVIVAL PREDICTION\n")

    df_raw = load_data()
    df_clean = preprocess(df_raw)
    model, X_train, X_test, y_train, y_test, features = train_model(df_clean)
    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, features)
    predict_passenger(model, features)

    print("\n✅ All done! Check confusion_matrix.png and feature_importance.png\n")
