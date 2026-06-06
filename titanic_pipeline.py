"""
简洁的 Titanic 生还预测流水线脚本。
整合 detect_outliers.py、set_missing_ages.py、plot_learning_curve.py 中的处理思路，
完成数据读取、清洗、特征工程、训练并输出 submission.csv。

用法:
    python titanic_pipeline.py

注意: 默认读取路径为 ../input/train.csv 和 ../input/test.csv（与原项目一致）。
"""

import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# reuse local helper modules if present
try:
    from detect_outliers import detect_outliers
except Exception:
    detect_outliers = None

try:
    from set_missing_ages import set_missing_ages
except Exception:
    set_missing_ages = None


def load_data(train_path="train.csv", test_path="test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def fix_cabin(df):
    df['Cabin'] = pd.Series([x[0] if not pd.isnull(x) else 'X' for x in df['Cabin']])


def basic_fill_missing(df):
    # 对象型用众数，数值型用中位数
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col in missing.index:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)


def extract_title(df):
    df['Title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
    newtitles = {
        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer", "Rev": "Officer",
        "Jonkheer": "Royalty", "Don": "Royalty", "Sir": "Royalty", "the Countess": "Royalty", "Dona": "Royalty", "Lady": "Royalty",
        "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs",
        "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master"
    }
    df['Title'] = df['Title'].map(newtitles).fillna('Rare')


def family_features(df):
    df['Fsize'] = df['SibSp'] + df['Parch'] + 1
    df['Single'] = (df['Fsize'] == 1).astype(int)
    df['SmallF'] = ((df['Fsize'] > 1) & (df['Fsize'] <= 4)).astype(int)
    df['LargeF'] = (df['Fsize'] > 4).astype(int)


def encode_features(df):
    # Cabin map
    cabin_map = {'X': 1, 'C': 2, 'B': 3, 'D': 4, 'E': 5, 'A': 6, 'F': 7, 'G': 8, 'T': 9}
    df['Cabin'] = df['Cabin'].map(cabin_map).fillna(1).astype(int)

    # Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Embarked
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
        df = pd.get_dummies(df, columns=['Embarked'], prefix='Em')

    # Pclass
    if 'Pclass' in df.columns:
        df['Pclass'] = df['Pclass'].astype('category')
        df = pd.get_dummies(df, columns=['Pclass'], prefix='Pc')

    # Title
    if 'Title' in df.columns:
        df = pd.get_dummies(df, columns=['Title'], prefix='T')

    return df


def preprocess(train, test, drop_outliers=True):
    # keep copies
    train = train.copy()
    test = test.copy()

    # drop rows with too many missing PassengerId etc. (not needed here)

    # detect and drop outliers in training set if function available
    if drop_outliers and (detect_outliers is not None):
        out_idx = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
        if out_idx:
            train = train.drop(out_idx, axis=0).reset_index(drop=True)

    # Fix cabin
    fix_cabin(train)
    fix_cabin(test)

    # Fill basics
    basic_fill_missing(train)
    basic_fill_missing(test)

    # Optionally use RandomForest to impute Age if helper is available
    if set_missing_ages is not None:
        try:
            train, rfr = set_missing_ages(train)
            # apply same rfr to test
            tmp = test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
            null_age = tmp[test.Age.isnull()].values
            if null_age.shape[0] > 0:
                X = null_age[:, 1:]
                predA = rfr.predict(X)
                test.loc[test.Age.isnull(), 'Age'] = predA
        except Exception:
            pass

    # Feature engineering
    extract_title(train)
    extract_title(test)
    family_features(train)
    family_features(test)

    # Create Ticket2 (length of ticket)
    train['Ticket2'] = train['Ticket'].apply(lambda x: len(str(x)))
    test['Ticket2'] = test['Ticket'].apply(lambda x: len(str(x)))

    # Encode features and get dummies
    train = encode_features(train)
    test = encode_features(test)

    # Drop unused columns
    drop_cols = ['Name', 'Ticket', 'PassengerId']
    for c in drop_cols:
        if c in train.columns:
            train.drop(c, axis=1, inplace=True)
        if c in test.columns:
            test.drop(c, axis=1, inplace=True)

    # Ensure train and test have same columns
    missing_cols = set(train.columns) - set(test.columns)
    for c in missing_cols:
        if c != 'Survived':
            test[c] = 0
    # align column order
    test = test[ [c for c in train.columns if c != 'Survived'] ]

    return train, test


def train_and_predict(train, test, output_path='submission.csv', passenger_ids=None):
    y = train['Survived']
    X = train.drop('Survived', axis=1)

    # simple pipeline
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    print('CV accuracy: mean={:.4f}, std={:.4f}'.format(scores.mean(), scores.std()))

    clf.fit(X, y)
    preds = clf.predict(test)

    if passenger_ids is None:
        try:
            old_test = pd.read_csv('../input/test.csv')
            passenger_ids = old_test['PassengerId']
        except Exception:
            passenger_ids = np.arange(1, len(preds) + 1)

    out = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds.astype(int)})
    out.to_csv(output_path, index=False)
    print('Saved predictions to', output_path)
    return out


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, '..', 'input', 'train.csv')
    test_path = os.path.join(base_dir, '..', 'input', 'test.csv')
    # fallback to relative input/ and then to files located in the script dir or cwd
    if not os.path.exists(train_path):
        # first try relative input/ (same as original fallback)
        rel_train = os.path.join('..', 'input', 'train.csv')
        rel_test = os.path.join('..', 'input', 'test.csv')
        if os.path.exists(rel_train):
            train_path = rel_train
            test_path = rel_test
        else:
            # try files next to this script
            local_train = os.path.join(base_dir, 'train.csv')
            local_test = os.path.join(base_dir, 'test.csv')
            if os.path.exists(local_train):
                train_path = local_train
                test_path = local_test
            else:
                # final fallback to plain names (cwd)
                train_path = 'train.csv'
                test_path = 'test.csv'

    print('Loading data...')
    train_df, test_df = load_data(train_path, test_path)
    print('Preprocessing...')
    train_p, test_p = preprocess(train_df, test_df)
    print('Training and predicting...')
    submission = train_and_predict(train_p, test_p, output_path='submission.csv', passenger_ids=test_df['PassengerId'])
    print(submission.head())
