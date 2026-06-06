# -*- coding: utf-8 -*-
"""
泰坦尼克号生还预测 — 基准版本
整合所有文件最佳技术 + 新增关键特征 + 加权Stacking集成
"""

import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from collections import Counter

# ============================================================================
# 1. 数据加载
# ============================================================================
print("=" * 60)
print("泰坦尼克号生还预测 — 基准版本")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
PassengerId = test['PassengerId'].copy()

# 保存原始数据用于后续处理
train_orig = train.copy()
test_orig = test.copy()

# ============================================================================
# 2. 辅助函数
# ============================================================================

def detect_outliers(df, n, features):
    """Tukey IQR 方法检测离群值"""
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > n]
    return multiple_outliers


def fill_age_rf(train_df, test_df):
    """使用 RandomForest 回归预测填充缺失年龄"""
    features_for_age = ['Pclass', 'SibSp', 'Parch', 'Fare']

    for df in [train_df, test_df]:
        known = df[df['Age'].notna()]
        unknown = df[df['Age'].isna()]

        if len(unknown) == 0:
            continue

        X_known = known[features_for_age].values
        y_known = known['Age'].values
        X_unknown = unknown[features_for_age].values

        rfr = RandomForestRegressor(n_estimators=200, random_state=42)
        rfr.fit(X_known, y_known)
        df.loc[df['Age'].isna(), 'Age'] = rfr.predict(X_unknown)

    return train_df, test_df


def target_encode_smooth(train_df, test_df, col, target='Survived', alpha=10):
    """贝叶斯平滑目标编码"""
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
    mapping = smooth.to_dict()
    train_df[col + '_TE'] = train_df[col].map(mapping)
    test_df[col + '_TE'] = test_df[col].map(mapping).fillna(global_mean)
    return train_df, test_df


# ============================================================================
# 3. 缺失值处理
# ============================================================================
print("\n[1/7] 处理缺失值...")

# Cabin → Deck
for df in [train, test]:
    df['Cabin_Letter'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'X')

# Embarked 众数填充
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])

# Fare 中位数填充
test['Fare'] = test['Fare'].fillna(train['Fare'].median())

# Age 用 RandomForest 回归预测填充
train, test = fill_age_rf(train, test)

# ============================================================================
# 4. 离群点剔除
# ============================================================================
print("[2/7] 剔除离群点...")

outlier_idx = detect_outliers(train, 1, ['Age', 'SibSp', 'Parch', 'Fare'])
print(f"  检测到 {len(outlier_idx)} 个离群点")
train = train.drop(outlier_idx, axis=0).reset_index(drop=True)

# ============================================================================
# 5. 特征工程
# ============================================================================
print("[3/7] 特征工程...")

for df in [train, test]:
    # --- 5.1 Title 提取 ---
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Capt': 'Officer',
        'Countess': 'Royalty', 'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Royalty',
        'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty', 'Mme': 'Mrs',
        'Sir': 'Royalty'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # --- 5.2 Family Size ---
    df['Fsize'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Fsize'] == 1).astype(int)

    def fsize_group(s):
        if s == 1: return 'Single'
        elif s <= 4: return 'SmallF'
        elif s <= 6: return 'MedF'
        else: return 'LargeF'
    df['Fsize_Group'] = df['Fsize'].apply(fsize_group)

    # --- 5.3 Ticket Frequency (同票人数) ---
    df['Ticket_Freq'] = df.groupby('Ticket')['Ticket'].transform('count')

    # --- 5.4 Ticket Prefix ---
    def get_ticket_prefix(ticket):
        ticket = str(ticket).replace('.', '').replace('/', '')
        parts = ticket.split()
        if len(parts) > 1:
            prefix = parts[0]
            if not prefix.isdigit():
                return prefix
        return 'NONE'
    df['Ticket_Prefix'] = df['Ticket'].apply(get_ticket_prefix)

    # --- 5.5 Ticket Length ---
    df['Ticket_Len'] = df['Ticket'].apply(lambda x: len(str(x)))

    # --- 5.6 Deck ---
    cabin_map = {'X': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    df['Cabin_Enc'] = df['Cabin_Letter'].map(cabin_map).fillna(0).astype(int)

    # --- 5.7 Name Length ---
    df['Name_Len'] = df['Name'].apply(len)

    # --- 5.8 Has Cabin ---
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    # --- 5.9 Age Group ---
    def age_group(age):
        if age <= 12: return 'Child'
        elif age <= 18: return 'Teenager'
        elif age <= 35: return 'Young'
        elif age <= 55: return 'Middle'
        else: return 'Old'
    df['Age_Group'] = df['Age'].apply(age_group)

    # --- 5.10 Is Child ---
    df['Is_Child'] = (df['Age'] <= 12).astype(int)

    # --- 5.11 Fare per Person ---
    df['Fare_Per'] = df['Fare'] / df['Fsize']

    # --- 5.12 Fare Log ---
    df['Fare_Log'] = np.log1p(df['Fare'])

    # --- 5.13 交互特征 ---
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    df['Fare_Pclass'] = df['Fare'] / (df['Pclass'] + 1)
    df['Sex_Age'] = df['Sex'] + '_' + df['Age_Group']

    # --- 5.14 Fare 分箱 ---
    df['Fare_Q'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')

    # --- 5.15 Age 分箱 ---
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 55, 80], labels=False, include_lowest=True)

    # --- 5.16 Pclass × Sex ---
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']

# --- Ticket Prefix 稀有合并 ---
prefix_counts = train['Ticket_Prefix'].value_counts()
rare_prefixes = prefix_counts[prefix_counts < 5].index
train['Ticket_Prefix'] = train['Ticket_Prefix'].replace(rare_prefixes, 'RARE')
test['Ticket_Prefix'] = test['Ticket_Prefix'].replace(rare_prefixes, 'RARE')
# 处理测试集中可能的新前缀
test['Ticket_Prefix'] = test['Ticket_Prefix'].apply(lambda x: x if x in train['Ticket_Prefix'].unique() else 'RARE')

# --- 目标编码 (平滑) ---
train, test = target_encode_smooth(train, test, 'Ticket_Prefix', 'Survived', alpha=15)
train, test = target_encode_smooth(train, test, 'Cabin_Letter', 'Survived', alpha=10)

# ============================================================================
# 6. 编码 & 标准化
# ============================================================================
print("[4/7] 编码特征...")

# 合并进行统一编码
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
train_clean = train.drop(columns=[c for c in drop_cols if c in train.columns])
test_clean = test.drop(columns=[c for c in drop_cols if c in test.columns])

# 将 Pclass 视为类别
train_clean['Pclass'] = train_clean['Pclass'].astype('object')
test_clean['Pclass'] = test_clean['Pclass'].astype('object')

combined = pd.concat([train_clean, test_clean], axis=0, ignore_index=True)

# 移除原始高基数字符串列（已用目标编码替代）
combined = combined.drop(columns=['Ticket_Prefix'], errors='ignore')

# One-Hot 编码
cat_cols = ['Sex', 'Embarked', 'Title', 'Fsize_Group', 'Pclass', 'Age_Group', 'Sex_Age', 'Cabin_Letter', 'Pclass_Sex']
cat_cols = [c for c in cat_cols if c in combined.columns]
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# 分离
train_proc = combined[:len(train_clean)].copy()
test_proc = combined[len(train_clean):].copy()

y = train['Survived'].values
X = train_proc.values
X_test = test_proc.values

# 标准化数值特征
scaler = StandardScaler()
num_feature_indices = []
num_features = ['Age', 'Fare', 'Age_Pclass', 'Fare_Pclass', 'Fare_Log',
                'Ticket_Prefix_TE', 'Cabin_Letter_TE', 'Fare_Per',
                'Name_Len', 'Ticket_Len', 'Ticket_Freq']
for feat in num_features:
    if feat in train_proc.columns:
        idx = list(train_proc.columns).index(feat)
        num_feature_indices.append(idx)

if num_feature_indices:
    X[:, num_feature_indices] = scaler.fit_transform(X[:, num_feature_indices])
    X_test[:, num_feature_indices] = scaler.transform(X_test[:, num_feature_indices])

feature_names = list(train_proc.columns)
print(f"  总特征数: {len(feature_names)}")

# ============================================================================
# 7. 超参数搜索
# ============================================================================
print("[5/7] 超参数搜索...")

# RandomizedSearchCV 对 XGB
param_xgb = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.05, 0.1]
}

cv_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

xgb_base = XGBClassifier(random_state=42, use_label_encoder=False,
                          eval_metric='logloss', verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_base, param_xgb, n_iter=30, cv=cv_inner,
    scoring='accuracy', n_jobs=1, verbose=0, random_state=42
)
xgb_search.fit(X, y)
best_xgb = xgb_search.best_estimator_
print(f"  XGB 最佳CV: {xgb_search.best_score_:.4f}")

# RandomizedSearchCV 对 LGBM
param_lgbm = {
    'n_estimators': [200, 300, 400, 500],
    'num_leaves': [15, 31, 50, 70],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_samples': [5, 10, 20],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

lgbm_base = LGBMClassifier(random_state=42, verbose=-1)
lgbm_search = RandomizedSearchCV(
    lgbm_base, param_lgbm, n_iter=30, cv=cv_inner,
    scoring='accuracy', n_jobs=1, verbose=0, random_state=42
)
lgbm_search.fit(X, y)
best_lgbm = lgbm_search.best_estimator_
print(f"  LGBM 最佳CV: {lgbm_search.best_score_:.4f}")

# RandomizedSearchCV 对 RF
param_rf = {
    'n_estimators': [300, 400, 500, 600],
    'max_depth': [8, 12, 15, 20, None],
    'min_samples_split': [2, 3, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_base = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(
    rf_base, param_rf, n_iter=20, cv=cv_inner,
    scoring='accuracy', n_jobs=1, verbose=0, random_state=42
)
rf_search.fit(X, y)
best_rf = rf_search.best_estimator_
print(f"  RF 最佳CV: {rf_search.best_score_:.4f}")

# ============================================================================
# 8. 加权 Stacking 集成
# ============================================================================
print("[6/7] 加权 Stacking 集成...")

models = [
    ('RF', best_rf),
    ('ET', ExtraTreesClassifier(n_estimators=500, max_depth=15,
                                min_samples_split=3, min_samples_leaf=2,
                                random_state=42)),
    ('XGB', best_xgb),
    ('LGBM', best_lgbm),
    ('GB', GradientBoostingClassifier(n_estimators=500, max_depth=6,
                                      learning_rate=0.03, subsample=0.8,
                                      random_state=42)),
    ('LR', LogisticRegression(solver='liblinear', C=0.8, random_state=42)),
    ('LDA', LinearDiscriminantAnalysis(solver='lsqr')),
]

# RepeatedStratifiedKFold for robust stacking
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
n_models = len(models)
n_repeats = 3
n_folds = 5

train_meta = np.zeros((X.shape[0], n_models))
test_meta = np.zeros((X_test.shape[0], n_models))
model_scores = []

for i, (name, model) in enumerate(models):
    print(f"  训练模型 {i+1}/{n_models}: {name}")
    test_meta_i = np.zeros((X_test.shape[0], n_folds * n_repeats))
    fold_idx = 0
    cv_scores = []

    for train_idx, val_idx in rskf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        train_meta[val_idx, i] = model.predict_proba(X_val)[:, 1]
        test_meta_i[:, fold_idx] = model.predict_proba(X_test)[:, 1]

        score = model.score(X_val, y_val)
        cv_scores.append(score)
        fold_idx += 1

    test_meta[:, i] = test_meta_i.mean(axis=1)
    avg_score = np.mean(cv_scores)
    model_scores.append(avg_score)
    print(f"    {name} CV: {avg_score:.4f}")

# 计算权重 (基于CV得分)
model_weights = np.array(model_scores)
model_weights = model_weights / model_weights.sum()
print(f"\n  模型权重: {dict(zip([m[0] for m in models], model_weights.round(4)))}")

# 加权融合
train_stack_weighted = train_meta * model_weights
test_stack_weighted = test_meta * model_weights

# 元模型 (Meta Model) — LGBM + 概率校准
meta_model = LGBMClassifier(
    n_estimators=300, num_leaves=31, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)

calibrator = CalibratedClassifierCV(estimator=meta_model, method='sigmoid', cv=5)
calibrator.fit(train_stack_weighted, y)

final_pred_proba = calibrator.predict_proba(test_stack_weighted)[:, 1]
final_predictions = (final_pred_proba >= 0.5).astype(int)

# ============================================================================
# 9. 输出结果
# ============================================================================
print("\n[7/7] 输出结果...")

# Stacking CV 评估
stack_cv = cross_val_score(
    LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    train_stack_weighted, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='accuracy'
)
print(f"  Stacking 元模型 CV: {stack_cv.mean():.4f} (+/- {stack_cv.std() * 2:.4f})")

# 单模型最佳得分
best_single_name = [m[0] for m in models][np.argmax(model_scores)]
best_single_score = max(model_scores)
print(f"  最佳单模型: {best_single_name} ({best_single_score:.4f})")

# 特征重要性
if hasattr(best_xgb, 'feature_importances_'):
    fi = pd.Series(best_xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\n  XGB 特征重要性 (前10):")
    for feat, imp in fi.head(10).items():
        print(f"    {feat}: {imp:.4f}")

if hasattr(best_lgbm, 'feature_importances_'):
    fi_lgb = pd.Series(best_lgbm.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\n  LGBM 特征重要性 (前10):")
    for feat, imp in fi_lgb.head(10).items():
        print(f"    {feat}: {imp:.4f}")

# 保存提交文件
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": final_predictions
})
submission.to_csv('submission_best.csv', index=False)

print("\n" + "=" * 60)
print(f"完成! 预测结果已保存至 submission_best.csv")
print(f"预测分布: 生还={sum(final_predictions==1)}, 遇难={sum(final_predictions==0)}")
print("=" * 60)
