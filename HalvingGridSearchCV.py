import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ================= 1. 数据预处理 =================
print("📊 正在加载数据...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 提取头衔（在填充年龄之前先提取，以便利用头衔分组填充年龄）
def extract_title(name):
    match = re.search(r' ([A-Za-z]+)\.', name)
    if match:
        return match.group(1)
    return "Unknown"

train["Title"] = train["Name"].apply(extract_title)
test["Title"] = test["Name"].apply(extract_title)

# 合并稀有头衔
title_mapping = {}
for title in train["Title"].unique():
    if train["Title"].value_counts()[title] < 10:
        title_mapping[title] = "Rare"
    else:
        title_mapping[title] = title

train["Title"] = train["Title"].map(title_mapping)
test["Title"] = test["Title"].map(title_mapping)

# 【优化1】按 Title 分组精准填充 Age 缺失值
print("✨ 正在按头衔分组精准填充 Age 缺失值...")
for dataset in [train, test]:
    age_by_title = dataset.groupby('Title')['Age'].median()
    dataset['Age'] = dataset.apply(lambda row: age_by_title[row['Title']] if pd.isna(row['Age']) else row['Age'], axis=1)

# 处理其他缺失值
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# 基础类别编码
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 头衔编码
title_encode = {title: idx for idx, title in enumerate(sorted(train["Title"].unique()))}
train["Title"] = train["Title"].map(title_encode)
test["Title"] = test["Title"].map(title_encode)

# 添加重要特征
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
train["FarePerPerson"] = train["Fare"] / train["FamilySize"]
test["FarePerPerson"] = test["Fare"] / test["FamilySize"]

# 【优化3】添加强交互特征
train['Pclass_Age'] = train['Pclass'] * train['Age']
test['Pclass_Age'] = test['Pclass'] * test['Age']

# 提取 Deck
def extract_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    return cabin[0]

train["Deck"] = train["Cabin"].apply(extract_deck)
test["Deck"] = test["Cabin"].apply(extract_deck)

# 提取 TicketPrefix
def extract_ticket_prefix(ticket):
    match = re.search(r'([A-Za-z\.\/]+)', str(ticket))
    if match:
        prefix = match.group(1).replace('.', '').replace('/', '').strip()
        return prefix if prefix else "None"
    return "None"

train["TicketPrefix"] = train["Ticket"].apply(extract_ticket_prefix)
test["TicketPrefix"] = test["Ticket"].apply(extract_ticket_prefix)

# 合并稀有前缀
prefix_counts = train["TicketPrefix"].value_counts()
prefix_map = {prefix: (prefix if count >= 10 else "Rare") for prefix, count in prefix_counts.items()}
train["TicketPrefix"] = train["TicketPrefix"].map(prefix_map)
test["TicketPrefix"] = test["TicketPrefix"].map(lambda x: prefix_map.get(x, "Rare"))

# 【优化2】目标编码 (Target Encoding) - 用存活率代替普通标签编码
# 计算训练集中各分类特征对应的 Survived 均值
def target_encode(train, test, col):
    target_mean = train.groupby(col)['Survived'].mean()
    train[col + '_target'] = train[col].map(target_mean)
    test[col + '_target'] = test[col].map(target_mean)
    # 填充测试集中可能出现的未见过的类别
    test[col + '_target'].fillna(train[col + '_target'].mean(), inplace=True)

# 对 Deck 和 TicketPrefix 使用目标编码
target_encode(train, test, 'Deck')
target_encode(train, test, 'TicketPrefix')

# 年龄和票价分箱
train["AgeBin"] = pd.cut(train["Age"], bins=[0, 12, 18, 35, 60, 80], labels=False)
test["AgeBin"] = pd.cut(test["Age"], bins=[0, 12, 18, 35, 60, 80], labels=False)
fare_bins = pd.qcut(train["Fare"], 4, retbins=True, duplicates="drop")[1]
train["FareBin"] = pd.cut(train["Fare"], bins=fare_bins, labels=False, include_lowest=True)
test["FareBin"] = pd.cut(test["Fare"], bins=fare_bins, labels=False, include_lowest=True)

# 选择特征（加入了交互特征和目标编码后的新特征）
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", 
            "FamilySize", "FarePerPerson", "AgeBin", "FareBin", "Pclass_Age", 
            "Deck_target", "TicketPrefix_target"]
            
X = train[features]
y = train["Survived"]
X_test = test[features]

# ================= 2. 设置进度条与调参 =================
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

print("🚀 正在启动智能调参（带进度显示）...")
print("⏳ 这可能需要几分钟，请观察下方进度...\n")

grid_search = HalvingGridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    factor=2, 
    cv=5, 
    verbose=2,  
    n_jobs=1, # 修改为 -1 充分利用你的 CPU 多核加速
    random_state=42
)

grid_search.fit(X, y)

# ================= 3. 交叉验证评估与结果提交 =================
print("\n✅ 调参完成！")
print("🔥 最佳参数组合：", grid_search.best_params_)

best_model = grid_search.best_estimator_

# 输出特征重要性
print("\n📊 正在计算特征重要性...")
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.title('随机森林特征重要性排序')
plt.gca().invert_yaxis()
plt.show()

# 交叉验证评估
print("\n🔄 正在进行 10 折分层交叉验证以评估模型真实性能...")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=kfold, scoring='accuracy')

print(f"✅ 交叉验证各折准确率: {cv_scores}")
print(f"📊 交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 训练最终模型并预测
print("\n🔄 正在用全量数据训练最终模型...")
best_model.fit(X, y)

print("🔮 正在预测测试集...")
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission_optimized_features.csv", index=False)
print("📝 submission_optimized_features.csv 已生成")