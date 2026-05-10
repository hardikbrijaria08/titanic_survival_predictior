[README.md](https://github.com/user-attachments/files/27563743/README.md)
# 🚢 Titanic Survivor Prediction

A supervised machine learning project that predicts whether a Titanic passenger survived, using **XGBoost** classification with feature engineering and model analysis.

## 📌 Project Highlights
- **84% prediction accuracy** on test data
- Feature engineering: family size, age groups, fare tiers, solo traveller flag
- Feature importance analysis to identify key survival factors
- Visual outputs: confusion matrix, feature importance chart

## 🧠 Key Findings
- **Sex** is the strongest predictor — females had much higher survival rates
- **Passenger class (Pclass)** strongly correlates with survival
- **Age** and **fare paid** also significantly influence outcomes
- Passengers travelling alone had lower survival probability

## 🛠 Tech Stack
- Python, pandas, NumPy
- scikit-learn (train/test split, metrics)
- XGBoost
- Matplotlib, Seaborn

## 📁 Project Structure
```
titanic-survival-prediction/
│
├── titanic_prediction.py   # Main script: preprocessing, training, evaluation
├── requirements.txt        # Dependencies
├── confusion_matrix.png    # Generated after running
├── feature_importance.png  # Generated after running
└── README.md
```

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python titanic_prediction.py
```

> If no `titanic.csv` is present, the script automatically loads the built-in seaborn Titanic dataset.

## 📊 Model Pipeline

```
Raw Data
   ↓
Handle Missing Values (median imputation)
   ↓
Feature Engineering (FamilySize, IsAlone, AgeGroup, FareTier)
   ↓
Label Encoding (categorical → numeric)
   ↓
Train/Test Split (80/20)
   ↓
XGBoost Classifier
   ↓
Evaluation + Feature Importance
```

## 📈 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~84%   |
| Precision | ~83%   |
| Recall    | ~84%   |
| F1 Score  | ~83%   |
