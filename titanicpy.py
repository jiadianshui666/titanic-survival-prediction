# -*- coding: utf-8 -*-
"""
泰坦尼克号生还预测项目
基于徐荣钦(深圳大学)项目报告复现
包含详细注释
优化版本：特征工程+模型调参+加权Stacking+GridSearchCV
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# --- 1. 数据加载 ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
PassengerId = test['PassengerId']

# --- 2. 数据预处理 ---
# 2.1 缺失值填补
train['Cabin'] = train['Cabin'].fillna('X')
test['Cabin'] = test['Cabin'].fillna('X')

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
test['Embarked'] = test['Embarked'].fillna(train['Embarked'].mode()[0])

test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])

def fill_age_rf(df):
    age_df = df[['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    if len(unknown_age) == 0:
        return df

    X = known_age[:, 1:]
    y = known_age[:, 0]

    rfr = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[df.Age.isnull(), 'Age'] = predictedAges
    return df

train = fill_age_rf(train)
test = fill_age_rf(test)

# 2.2 离群点剔除
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = pd.Series(outlier_indices)
    multiple_outliers = outlier_indices.value_counts() > n
    return list(multiple_outliers[multiple_outliers].index)

Outliers_to_drop = detect_outliers(train, 1, ['Age', 'SibSp', 'Parch', 'Fare'])
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)

# --- 3. 特征工程 (优化版) ---
# 3.1 Name -> Title
def get_title(df):
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
        'Countess': 'Royalty', 'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Royalty',
        'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty', 'Mme': 'Mrs',
        'Capt': 'Officer', 'Sir': 'Royalty'
    }
    df['Title'] = df['Title'].map(title_map)
    return df

train = get_title(train)
test = get_title(test)

# 3.2 Family Size
train['Fsize'] = train['SibSp'] + train['Parch'] + 1
test['Fsize'] = test['SibSp'] + test['Parch'] + 1

def fsize_group(size):
    if size == 1:
        return 'Single'
    elif size <= 4:
        return 'SmallF'
    elif size <= 6:
        return 'MedF'
    else:
        return 'LargeF'

train['Fsize_Group'] = train['Fsize'].apply(fsize_group)
test['Fsize_Group'] = test['Fsize'].apply(fsize_group)

# 3.3 Ticket Length
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))

# 3.4 Cabin首字母
train['Cabin'] = train['Cabin'].str[0]
test['Cabin'] = test['Cabin'].str[0]

cabin_map = {'X':0, 'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8}
train['Cabin'] = train['Cabin'].map(cabin_map)
test['Cabin'] = test['Cabin'].map(cabin_map)

# --- 4. 新增特征工程 ---
# 4.1 Is_Alone: 是否独自一人
train['Is_Alone'] = (train['Fsize'] == 1).astype(int)
test['Is_Alone'] = (test['Fsize'] == 1).astype(int)

# 4.2 Is_Child: 是否为儿童 (Age <= 12)
train['Is_Child'] = (train['Age'] <= 12).astype(int)
test['Is_Child'] = (test['Age'] <= 12).astype(int)

# 4.3 Age分组
def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 35:
        return 'Young'
    elif age <= 55:
        return 'Middle'
    else:
        return 'Old'

train['Age_Group'] = train['Age'].apply(age_group)
test['Age_Group'] = test['Age'].apply(age_group)

# 4.4 Fare分箱 (使用训练集的分位数边界)
fare_q = train['Fare'].quantile([0, 0.25, 0.5, 0.75, 1]).values
train['Fare_Group'] = pd.cut(train['Fare'], bins=fare_q, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
test['Fare_Group'] = pd.cut(test['Fare'], bins=fare_q, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)

# 4.5 交互特征
train['Age_Pclass'] = train['Age'] * train['Pclass']
test['Age_Pclass'] = test['Age'] * test['Pclass']

train['Fare_Pclass'] = train['Fare'] / (train['Pclass'] + 1)
test['Fare_Pclass'] = test['Fare'] / (test['Pclass'] + 1)

# --- 5. 高级特征工程 ---
# 5.1 性别与年龄的交互
train['Sex_Age'] = train['Sex'] + '_' + train['Age_Group']
test['Sex_Age'] = test['Sex'] + '_' + test['Age_Group']

# 5.2 Ticket前缀
def get_ticket_prefix(ticket):
    ticket = ticket.replace('.', '').replace('/', '')
    parts = ticket.split()
    if len(parts) > 1:
        return parts[0]
    else:
        return 'NONE'

train['Ticket_Prefix'] = train['Ticket'].apply(get_ticket_prefix)
test['Ticket_Prefix'] = test['Ticket'].apply(get_ticket_prefix)

# 5.5 对高基数类别 Ticket_Prefix 使用目标编码（平滑）
def target_encode_smooth(train_df, test_df, col, target, alpha=10):
    global_mean = train_df[target].mean()
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    smooth = (agg['mean'] * agg['count'] + global_mean * alpha) / (agg['count'] + alpha)
    mapping = smooth.to_dict()
    train_df[col + '_TE'] = train_df[col].map(mapping)
    test_df[col + '_TE'] = test_df[col].map(mapping).fillna(global_mean)
    return train_df, test_df

train, test = target_encode_smooth(train, test, 'Ticket_Prefix', 'Survived', alpha=15)

# 将 Ticket_Len 分箱并进行目标编码（在 target_encode_smooth 定义之后调用）
train['Ticket_Len_Bin'] = pd.qcut(train['Ticket_Len'], 4, labels=False, duplicates='drop')
test['Ticket_Len_Bin'] = pd.qcut(test['Ticket_Len'], 4, labels=False, duplicates='drop')
train, test = target_encode_smooth(train, test, 'Ticket_Len_Bin', 'Survived', alpha=8)

# 5.3 Fare的对数变换
train['Fare_Log'] = np.log1p(train['Fare'])
test['Fare_Log'] = np.log1p(test['Fare'])

# 5.4 Cabin甲板信息
def get_deck(cabin):
    if cabin == 0:
        return 'U'
    else:
        return list(cabin_map.keys())[list(cabin_map.values()).index(cabin)]

train['Deck'] = train['Cabin'].apply(get_deck)
test['Deck'] = test['Cabin'].apply(get_deck)

# --- 6. 数据清洗与编码 ---
drop_columns = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Fsize']
train_clean = train.drop(drop_columns, axis=1)
test_clean = test.drop(drop_columns, axis=1)

train_clean['Pclass'] = train_clean['Pclass'].astype('object')
test_clean['Pclass'] = test_clean['Pclass'].astype('object')

combined = pd.concat([train_clean, test_clean], axis=0)
# 移除原始高基数字符串列，使用目标编码列 Ticket_Prefix_TE
combined = combined.drop(columns=['Ticket_Prefix'], errors='ignore')
# 使用目标编码后的 Ticket_Prefix_TE，故从独热编码列中移除 'Ticket_Prefix'
combined = pd.get_dummies(combined, columns=['Sex', 'Embarked', 'Title', 'Fsize_Group', 'Pclass', 'Age_Group', 'Fare_Group', 'Sex_Age', 'Deck'], drop_first=True)

train_proc = combined[:len(train_clean)]
test_proc = combined[len(train_clean):]
test_proc = test_proc.drop(columns=['Survived'], errors='ignore')

y = train_proc['Survived']
X = train_proc.drop('Survived', axis=1)

# --- 7. 特征标准化 ---
scaler = StandardScaler()
# 将 Ticket_Prefix_TE 和 Ticket_Len_Bin_TE 包含进标准化的数值特征
num_cols = ['Age', 'Fare', 'Age_Pclass', 'Fare_Pclass', 'Fare_Log']
if 'Ticket_Prefix_TE' in X.columns:
    num_cols.append('Ticket_Prefix_TE')
if 'Ticket_Len_Bin_TE' in X.columns:
    num_cols.append('Ticket_Len_Bin_TE')
X[num_cols] = scaler.fit_transform(X[num_cols])
test_proc[num_cols] = scaler.transform(test_proc[num_cols])

# --- 8. 超参数搜索 (RandomizedSearchCV 扩展) ---
print("="*60)
print("开始超参数随机搜索 (XGB & LGBM)...")
print("="*60)

# 定义随机搜索的参数分布
param_dists = {
    'XGB': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
    },
    'LGBM': {
        'n_estimators': [200, 300, 400, 500],
        'num_leaves': [15, 31, 40, 60],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# 搜索最优参数的模型
best_models = {}

param_dists['RF'] = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [8, 12, 15, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 4]
}

for model_name in ['XGB', 'LGBM', 'RF']:
    print(f"\n正在随机搜索 {model_name} 的最优参数...")
    if model_name == 'XGB':
        base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    else:
        base_model = LGBMClassifier(random_state=42, verbose=-1)

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    rand_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dists[model_name],
        n_iter=24,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    rand_search.fit(X, y)
    best_models[model_name] = rand_search.best_estimator_
    print(f"{model_name} 最优参数: {rand_search.best_params_}")
    print(f"{model_name} 最优CV得分: {rand_search.best_score_:.4f}")

# 其他模型使用预设参数
best_models['RF'] = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=42)

print("\n" + "="*60)
print("超参数搜索完成！")
print("="*60)

# 8.1 输出特征重要性（基于 XGB / LGBM）
def print_feature_importances(model, name):
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\n{name} 特征重要性 (前20):")
        print(fi.head(20))
        fi.to_csv(f'feature_importance_{name}.csv')
    else:
        print(f"{name} 不支持 feature_importances_")

print_feature_importances(best_models['XGB'], 'XGB')
print_feature_importances(best_models['LGBM'], 'LGBM')

# --- 9. 模型构建与加权Stacking ---
# 使用搜索到的最优参数构建基模型
print("\n开始集成模型训练...")

models = [
    ('RF', best_models['RF']),
    ('ET', ExtraTreesClassifier(n_estimators=500, max_depth=15, min_samples_split=3, min_samples_leaf=2, random_state=42)),
    ('XGB', best_models['XGB']),
    ('LGBM', best_models['LGBM']),
    ('GB', GradientBoostingClassifier(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42)),
    ('LDA', LinearDiscriminantAnalysis(solver='lsqr')),
    ('LogReg', LogisticRegression(solver='liblinear', random_state=42, C=0.8)),
]

# 使用RepeatedStratifiedKFold进行交叉验证
def weighted_stacking(models, train, test, y, n_folds=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)

    train_meta = np.zeros((train.shape[0], len(models)))
    test_meta = np.zeros((test.shape[0], len(models)))
    model_weights = np.zeros(len(models))
    model_scores = []

    for i, (name, model) in enumerate(models):
        print(f"\nTraining Model {i+1}/{len(models)}: {name}")
        test_meta_i = np.zeros((test.shape[0], n_folds * n_repeats))
        fold_idx = 0
        cv_scores = []

        for train_idx, val_idx in rskf.split(train, y):
            X_tr, X_val = train.iloc[train_idx], train.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            train_meta[val_idx, i] = model.predict_proba(X_val)[:, 1]
            test_meta_i[:, fold_idx] = model.predict_proba(test)[:, 1]

            score = model.score(X_val, y_val)
            cv_scores.append(score)
            fold_idx += 1

        test_meta[:, i] = test_meta_i.mean(axis=1)
        avg_score = np.mean(cv_scores)
        model_scores.append(avg_score)
        print(f"  {name} CV Score: {avg_score:.4f}")

    model_weights = np.array(model_scores)
    model_weights = model_weights / model_weights.sum()
    print(f"\n模型权重: {dict(zip([m[0] for m in models], model_weights.round(3)))}")

    return train_meta, test_meta, model_weights

train_stack, test_stack, weights = weighted_stacking(models, X, test_proc, y, n_folds=5, n_repeats=3)

# 加权融合基模型预测
train_stack_weighted = train_stack * weights
test_stack_weighted = test_stack * weights

# 第二层模型 (Meta Model)
meta_model = LGBMClassifier(
    n_estimators=300,
    num_leaves=31,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# 对元模型进行概率校准
calibrator = CalibratedClassifierCV(estimator=meta_model, method='sigmoid', cv=5)
calibrator.fit(train_stack_weighted, y)

# 最终预测（经过校准）
final_pred_meta = calibrator.predict(test_stack_weighted)
final_predictions = final_pred_meta

# --- 10. 结果输出 ---
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": final_predictions
})
submission.to_csv('submission.csv', index=False)
print("\n" + "="*60)
print("预测完成，结果已保存至 submission.csv")
print("="*60)
