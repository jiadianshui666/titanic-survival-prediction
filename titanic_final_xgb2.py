# -*- coding: utf-8 -*-
"""
Titanic Final - XGB_v2_sigmoid
基于 titanic_best.py 改进:
1. 软截断异常值 (替代硬删除)
2. 新增特征 (70个 vs 59个)
3. XGB_v2 作为元模型 (替代 LGBM)
4. 阈值优化 (网格搜索确定)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 数据加载
# ============================================================================
print("=" * 60)
print("Titanic Final - XGB_v2_sigmoid")
print("=" * 60)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
PassengerId = test['PassengerId'].copy()

# ============================================================================
# 预处理函数
# ============================================================================

def cap_outliers(df, features, multiplier=1.5):
    """软异常值处理: 截断而非删除"""
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def fill_age_rf(train_df, test_df):
    """使用 RandomForest 回归预测填充缺失年龄"""
    features_for_age = ['Pclass', 'SibSp', 'Parch', 'Fare']
    for df in [train_df, test_df]:
        known = df[df['Age'].notna()]
        unknown = df[df['Age'].isna()]
        if len(unknown) == 0:
            continue
        rfr = RandomForestRegressor(n_estimators=200, random_state=42)
        rfr.fit(known[features_for_age].values, known['Age'].values)
        df.loc[df['Age'].isna(), 'Age'] = rfr.predict(unknown[features_for_age].values)
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
# 数据预处理
# ============================================================================
print("\n[1/5] 数据预处理...")

for df in [train, test]:
    df['Cabin_Letter'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'X')

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])
test['Fare'] = test['Fare'].fillna(train['Fare'].median())
train, test = fill_age_rf(train, test)

# 软截断异常值 (不删除样本)
train = cap_outliers(train, ['Age', 'SibSp', 'Parch', 'Fare'], multiplier=2.0)
print("  异常值已截断，保留全部样本")

# ============================================================================
# 特征工程
# ============================================================================
print("[2/5] 特征工程...")

for df in [train, test]:
    # Title
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                 'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
                 'Capt': 'Officer', 'Countess': 'Royalty', 'Mlle': 'Miss', 'Ms': 'Miss',
                 'Lady': 'Royalty', 'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty',
                 'Mme': 'Mrs', 'Sir': 'Royalty'}
    df['Title'] = df['Title'].map(title_map).fillna('Rare')

    # Family Size
    df['Fsize'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Fsize'] == 1).astype(int)
    df['Fsize_Group'] = df['Fsize'].apply(lambda s: 'Single' if s == 1 else ('SmallF' if s <= 4 else ('MedF' if s <= 6 else 'LargeF')))

    # Ticket
    df['Ticket_Freq'] = df.groupby('Ticket')['Ticket'].transform('count')
    def get_ticket_prefix(ticket):
        ticket = str(ticket).replace('.', '').replace('/', '')
        parts = ticket.split()
        if len(parts) > 1 and not parts[0].isdigit():
            return parts[0]
        return 'NONE'
    df['Ticket_Prefix'] = df['Ticket'].apply(get_ticket_prefix)
    df['Ticket_Len'] = df['Ticket'].apply(lambda x: len(str(x)))

    # Cabin
    cabin_map = {'X': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
    df['Cabin_Enc'] = df['Cabin_Letter'].map(cabin_map).fillna(0).astype(int)
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df['Cabin_Info'] = df['Cabin'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # Name
    df['Name_Len'] = df['Name'].apply(len)

    # Age
    df['Age_Group'] = df['Age'].apply(lambda x: 'Child' if x <= 12 else ('Teenager' if x <= 18 else ('Young' if x <= 35 else ('Middle' if x <= 55 else 'Old'))))
    df['Is_Child'] = (df['Age'] <= 12).astype(int)
    df['Is_Elderly'] = (df['Age'] >= 60).astype(int)

    # Fare
    df['Fare_Per'] = df['Fare'] / df['Fsize']
    df['Fare_Log'] = np.log1p(df['Fare'])
    df['Fare_Q'] = pd.qcut(df['Fare'], 4, labels=False, duplicates='drop')

    # 交互特征
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    df['Fare_Pclass'] = df['Fare'] / (df['Pclass'] + 1)
    df['Sex_Age'] = df['Sex'] + '_' + df['Age_Group']
    df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 55, 80], labels=False, include_lowest=True)
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']

    # 新增特征
    df['Rich_Woman'] = ((df['Sex'] == 'female') & (df['Fare'] > df['Fare'].median())).astype(int)
    df['Poor_Man'] = ((df['Sex'] == 'male') & (df['Fare'] < df['Fare'].median())).astype(int)
    df['Child_HighClass'] = ((df['Age'] <= 12) & (df['Pclass'] == 1)).astype(int)
    df['Family_Survival'] = ((df['Ticket_Freq'] > 1) & (df['Sex'] == 'female')).astype(int)
    df['Fare_Age_Ratio'] = df['Fare'] / (df['Age'] + 1)
    df['Has_SibSp'] = (df['SibSp'] > 0).astype(int)
    df['Has_Parch'] = (df['Parch'] > 0).astype(int)
    df['Ticket_Freq_Log'] = np.log1p(df['Ticket_Freq'])

# Ticket Prefix 稀有合并
prefix_counts = train['Ticket_Prefix'].value_counts()
rare_prefixes = prefix_counts[prefix_counts < 5].index
train['Ticket_Prefix'] = train['Ticket_Prefix'].replace(rare_prefixes, 'RARE')
test['Ticket_Prefix'] = test['Ticket_Prefix'].replace(rare_prefixes, 'RARE')
test['Ticket_Prefix'] = test['Ticket_Prefix'].apply(lambda x: x if x in train['Ticket_Prefix'].unique() else 'RARE')

# 目标编码
train, test = target_encode_smooth(train, test, 'Ticket_Prefix', 'Survived', alpha=15)
train, test = target_encode_smooth(train, test, 'Cabin_Letter', 'Survived', alpha=10)
train, test = target_encode_smooth(train, test, 'Title', 'Survived', alpha=10)

# ============================================================================
# 编码
# ============================================================================
print("[3/5] 编码特征...")

drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
train_clean = train.drop(columns=[c for c in drop_cols if c in train.columns])
test_clean = test.drop(columns=[c for c in drop_cols if c in test.columns])
train_clean['Pclass'] = train_clean['Pclass'].astype('object')
test_clean['Pclass'] = test_clean['Pclass'].astype('object')

combined = pd.concat([train_clean, test_clean], axis=0, ignore_index=True)
combined = combined.drop(columns=['Ticket_Prefix'], errors='ignore')

cat_cols = ['Sex', 'Embarked', 'Title', 'Fsize_Group', 'Pclass', 'Age_Group', 'Sex_Age', 'Cabin_Letter', 'Pclass_Sex']
cat_cols = [c for c in cat_cols if c in combined.columns]
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

train_proc = combined[:len(train_clean)].copy()
test_proc = combined[len(train_clean):].copy()

y = train['Survived'].values
X = train_proc.values
X_test = test_proc.values

scaler = StandardScaler()
num_features = ['Age', 'Fare', 'Age_Pclass', 'Fare_Pclass', 'Fare_Log',
                'Ticket_Prefix_TE', 'Cabin_Letter_TE', 'Title_TE', 'Fare_Per',
                'Name_Len', 'Ticket_Len', 'Ticket_Freq', 'Fare_Age_Ratio', 'Ticket_Freq_Log']
num_feature_indices = []
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
# 模型训练
# ============================================================================
print("[4/5] 模型训练...")

base_models = [
    ('RF', RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=42)),
    ('ET', ExtraTreesClassifier(n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=42)),
    ('XGB', XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)),
    ('LGBM', LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)),
    ('CatBoost', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, random_state=42, verbose=0)),
    ('GB', GradientBoostingClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42)),
]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
n_models = len(base_models)
train_meta = np.zeros((X.shape[0], n_models))
test_meta = np.zeros((X_test.shape[0], n_models))
model_scores = []

for i, (name, model) in enumerate(base_models):
    test_meta_i = np.zeros((X_test.shape[0], 15))
    fold_idx = 0
    cv_scores = []
    for train_idx, val_idx in rskf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        train_meta[val_idx, i] = model.predict_proba(X_val)[:, 1]
        test_meta_i[:, fold_idx] = model.predict_proba(X_test)[:, 1]
        cv_scores.append(model.score(X_val, y_val))
        fold_idx += 1
    test_meta[:, i] = test_meta_i.mean(axis=1)
    model_scores.append(np.mean(cv_scores))
    print(f"  {name}: {model_scores[-1]:.4f}")

# 加权融合
model_weights = np.array(model_scores)
model_weights = model_weights / model_weights.sum()
train_stack = train_meta * model_weights
test_stack = test_meta * model_weights

# ============================================================================
# 元模型: XGB_v2 + Sigmoid 校准
# ============================================================================
print("[5/5] 元模型 + 预测...")

meta_xgb2 = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, random_state=42,
    use_label_encoder=False, eval_metric='logloss', verbosity=0
)
calibrator = CalibratedClassifierCV(estimator=meta_xgb2, method='sigmoid', cv=5)
calibrator.fit(train_stack, y)
final_proba = calibrator.predict_proba(test_stack)[:, 1]

# 最优阈值 0.45
threshold = 0.45
final_predictions = (final_proba >= threshold).astype(int)

# 保存
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': final_predictions})
submission.to_csv('submission_XGB_v2_sigmoid.csv', index=False)

print(f"\n  阈值: {threshold}")
print(f"  生还: {sum(final_predictions==1)}, 遇难: {sum(final_predictions==0)}")
print(f"\n  保存至 submission_XGB_v2_sigmoid.csv")
print("=" * 60)
