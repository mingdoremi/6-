import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
import pickle
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# 난수 고정
np.random.seed(42)
random.seed(42)
# 한글 폰트 설정 (MacOS)
matplotlib.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
st.title(":시험관: 랜덤포레스트 + 하이퍼파라미터 튜닝 불량 여부 분류기")
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/minseo/Documents/Streamlit/실전 프로젝트/반도체_결함_데이터_한글.csv")
    df = df.dropna(subset=["불량여부"])
    df = df.sort_index()  # 인덱스 정렬
    return df
df = load_data()
X = df.drop(columns=['불량여부', '결함유형']).sort_index()
y = df['불량여부'].map({'FALSE': 0, 'REAL': 1}).sort_index()
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)
MODEL_PATH = "randomforest_defect_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.info("✅: 저장된 최적 모델을 불러왔습니다.")
else:
    st.warning(":로켓: 모델을 새로 학습합니다. (하이퍼파라미터 탐색 포함)")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    st.write("Best Params:", grid_search.best_params_)
    st.write(f"Best CV Score (recall): {grid_search.best_score_:.6f}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    st.success(":불: 최적 모델을 저장했습니다.")
y_proba = model.predict_proba(X_test)[:, 1]
threshold = st.slider("Threshold 값 설정 (불량으로 예측할 최소 확률)", 0.0, 1.0, 0.5, 0.01)
y_pred = (y_proba >= threshold).astype(int)
# 인덱스 맞추기
y_pred = pd.Series(y_pred, index=y_test.index)
st.subheader("📋 분류 리포트")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(6)
st.dataframe(report_df)
st.subheader("🔍 혼동 행렬 (Confusion Matrix)")
fig_cm, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("예측값")
ax.set_ylabel("실제값")
st.pyplot(fig_cm)
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
fig_roc, ax = plt.subplots(figsize=(6, 4))
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.6f}")
ax.plot([0, 1], [0, 1], '--', color='gray')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig_roc)
st.subheader(":압정: 주요 특성 중요도 (Top 20)")
importances = model.feature_importances_
features = X_encoded.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)
fig_imp, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=feat_importance[:20], y=feat_importance.index[:20], ax=ax)
ax.set_title("Top 20 Feature Importances")
st.pyplot(fig_imp)