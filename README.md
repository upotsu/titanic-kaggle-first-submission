# Titanic Kaggle First Submission

Kaggle の入門コンペ「Titanic - Machine Learning from Disaster」に取り組み、  
乗客データから生存予測モデルを作成し、提出用 CSV を生成した学習記録です。

## Qiita 
https://qiita.com/dorayaki800/items/d8a10174cbafc96bfa4c

## 目的

Kaggle の基本的な流れを理解することを目的としています。

- データセットの読み込み
- データ構造の確認
- 欠損値の確認
- 特徴量の選択
- 欠損値補完
- カテゴリ変数の数値化
- 機械学習モデルの学習
- 検証データでの精度確認
- 提出用 CSV の作成

## 使用技術

- Python
- pandas
- scikit-learn
- Kaggle Notebook
- RandomForestClassifier

## データセット

Kaggle Competition: Titanic - Machine Learning from Disaster

使用ファイル:

- train.csv
- test.csv
- gender_submission.csv

※ Kaggle のデータセットはリポジトリには含めず、Kaggle Notebook 上で読み込んでいます。

## 実施内容

### 1. データ読み込み

```python
import pandas as pd

train = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")
