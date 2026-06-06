# 🚢 Kaggle Titanic Survival Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Titanic-blue)](https://www.kaggle.com/c/titanic)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Kaggle 泰坦尼克号生存预测项目，完整的机器学习实战案例

## 📊 项目简介

基于泰坦尼克号乘客数据（891条训练样本），构建机器学习模型预测乘客生还情况。项目完成了从数据清洗到模型部署的全流程，涵盖特征工程、模型集成、超参数调优等核心技能。

## 🛠️ 技术栈

- **Python 3.8+**
- **Scikit-learn** - 机器学习框架
- **XGBoost** - 梯度提升算法
- **LightGBM** - 轻量级梯度提升
- **CatBoost** - 类别特征提升
- **Pandas** - 数据处理
- **NumPy** - 数值计算

## 📁 项目结构

```
Titanic/
├── README.md                      # 项目说明
├── requirements.txt               # 依赖包
├── LICENSE                        # 开源协议
├── titanic_final_xgb2.py          # 最终版本脚本
├── titanic_best.py                # 原版基准脚本
├── titanic_pipeline.py            # Pipeline 版本
├── HalvingGridSearchCV.py         # 超参数调优脚本
├── detect_outliers.py             # 异常值检测脚本
├── set_missing_ages.py            # 年龄填充脚本
├── train.csv                      # 训练数据
├── test.csv                       # 测试数据
└── submission_XGB_v2_sigmoid.csv  # 最终提交文件
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行最终版本

```bash
python titanic_final_xgb2.py
```

### 4. 查看结果

生成的预测文件：`submission_XGB_v2_sigmoid.csv`

## 📈 模型架构

### 数据预处理
- **异常值处理**：Tukey IQR 方法检测，软截断保留全部样本
- **缺失值填充**：RandomForest 回归预测填充年龄，众数填充登船港口

### 特征工程（70个特征）
- **基础特征**：家庭规模、船票频率、船舱等级、年龄分组
- **交互特征**：性别×年龄、票价×舱位、女性高票价
- **目标编码**：贝叶斯平滑处理高基数类别特征

### 模型集成
- **基模型**：RandomForest、ExtraTrees、XGBoost、LightGBM、CatBoost、GBDT
- **集成策略**：加权 Stacking，基于交叉验证得分分配权重
- **元模型**：XGBoost + Sigmoid 概率校准
- **阈值优化**：网格搜索确定最优分类阈值

## 📊 模型性能

| 模型 | 说明 |
|------|------|
| RandomForest | 基于 Bagging 的集成学习 |
| CatBoost | 类别特征提升算法 |
| ExtraTrees | 极端随机树 |
| XGBoost | 梯度提升算法 |
| LightGBM | 轻量级梯度提升 |
| GBDT | 梯度提升决策树 |

> 采用 5 折交叉验证 + 3 次重复评估模型性能

## 🔧 脚本说明

| 脚本 | 用途 |
|------|------|
| `titanic_final_xgb2.py` | 最终版本，生成提交文件 |
| `titanic_best.py` | 原版基准，作为对比 |
| `titanic_pipeline.py` | Pipeline 封装版本 |
| `HalvingGridSearchCV.py` | 超参数调优脚本 |
| `detect_outliers.py` | 异常值检测可视化 |
| `set_missing_ages.py` | 年龄填充方法对比 |

## 📝 面试问题

<details>
<summary>为什么用软截断而不是删除异常值？</summary>
保留样本量，避免信息丢失。删除异常值会减少训练数据，对小数据集影响较大。
</details>

<details>
<summary>Stacking 的原理是什么？</summary>
基模型输出作为元模型输入，降低方差，提升泛化能力。
</details>

<details>
<summary>为什么选 XGBoost 做元模型？</summary>
正则化能力强，防止过拟合，且对不平衡数据表现良好。
</details>

<details>
<summary>分类阈值是怎么确定的？</summary>
网格搜索测试不同阈值，选择交叉验证得分最高的。
</details>

## 📚 参考资料

- [Kaggle Titanic 竞赛](https://www.kaggle.com/c/titanic)
- [Scikit-learn 官方文档](https://scikit-learn.org/)
- [XGBoost 官方文档](https://xgboost.readthedocs.io/)
- [LightGBM 官方文档](https://lightgbm.readthedocs.io/)

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- Kaggle 提供数据集和竞赛平台
- 开源社区提供优秀的机器学习工具

---

⭐ 如果这个项目对你有帮助，请给个 Star！
