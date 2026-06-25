import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import xgboost as xgb

# ==================== 1. LOAD DATA ====================
df = pd.read_csv("GermanCredit.csv")
df['target'] = df['credit_risk']

print("Data Shape:", df.shape)
print("\nTarget distribution:\n", df['target'].value_counts(normalize=True) * 100)

# ==================== 2. DEFINE FEATURES MANUALLY (Important!) ====================
feature_columns = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                   'savings', 'employment_duration', 'installment_rate',
                   'personal_status_sex', 'property', 'age', 'housing', 'job',
                   'other_debtors', 'other_installment_plans']

# Explicitly define which are categorical and numerical
categorical_cols = ['status', 'credit_history', 'purpose', 'savings',
                   'employment_duration', 'personal_status_sex', 'property',
                   'housing', 'job', 'other_debtors', 'other_installment_plans']

numerical_cols = ['duration', 'amount', 'installment_rate', 'age']

X = df[feature_columns]
y = df['target']

# ==================== 3. SPLIT & PREPROCESS ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# ==================== 4. MODEL (XGBoost) ====================
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=2.3,      # Helps with 70:30 imbalance
        random_state=42,
        eval_metric='auc'
    ))
])

model.fit(X_train, y_train)

# ==================== 5. EVALUATE ====================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred))
print("AUC Score:", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
importances = model.named_steps['classifier'].feature_importances_
feat_names = (model.named_steps['preprocessor']
              .get_feature_names_out())
top_features = pd.Series(importances, index=feat_names).nlargest(10)
top_features.plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Most Important Features')
plt.show()

joblib.dump(model, 'credit_risk_model.pkl')
print("\n✅ Model saved successfully!")