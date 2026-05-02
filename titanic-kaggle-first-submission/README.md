# Titanic Kaggle First Submission

Kaggle の入門コンペ「Titanic - Machine Learning from Disaster」に取り組み、  
乗客データから生存予測モデルを作成し、提出用 CSV を生成した学習記録です。

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

- `train.csv`
- `test.csv`
- `gender_submission.csv`

Kaggle のデータセットはリポジトリには含めず、Kaggle Notebook 上で読み込む想定です。

## ディレクトリ構成

```text
titanic-kaggle-first-submission/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ titanic_first_submission.ipynb
├─ src/
│  └─ titanic_baseline.py
└─ submissions/
   └─ .gitkeep
```

## 実施内容

### 1. データ読み込み

```python
import pandas as pd

train = pd.read_csv("/kaggle/input/competitions/titanic/train.csv")
test = pd.read_csv("/kaggle/input/competitions/titanic/test.csv")
```

### 2. データ確認

```python
train.info()
train.isnull().sum()
```

確認した欠損値:

| カラム | 欠損数 |
|---|---:|
| Age | 177 |
| Cabin | 687 |
| Embarked | 2 |

### 3. 使用した特徴量

```python
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
```

初回モデルでは、以下のカラムは使用していません。

- `PassengerId`
- `Name`
- `Ticket`
- `Cabin`

### 4. 前処理

- `Age` は中央値で補完
- `Embarked` は最頻値で補完
- `Fare` は test 側の欠損対策として中央値で補完
- `Sex` と `Embarked` は one-hot encoding で数値化
- `train` 側と `test` 側の列をそろえる

### 5. モデル学習

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

valid_predictions = model.predict(X_valid)
accuracy = accuracy_score(y_valid, valid_predictions)

print("Validation Accuracy:", accuracy)
```

## 結果

検証データで以下の精度を確認しました。

```text
Validation Accuracy: 0.8044692737430168
```

約 80.4% の正解率となりました。

## 実行方法

Kaggle Notebook 上で実行する場合は、Titanic コンペのデータを Notebook に追加した上で、以下を実行します。

```bash
python src/titanic_baseline.py
```

実行後、ルートディレクトリに `submission.csv` が作成されます。

## 今後の改善案

- `Name` から敬称を抽出する
- `FamilySize` を作成する
- `IsAlone` を作成する
- `Cabin` の有無を特徴量化する
- `Age` の補完方法を改善する
- Logistic Regression や XGBoost など他モデルと比較する
- Cross Validation を導入する
- Kaggle の Public Score と Validation Accuracy の差を確認する

## 学習メモ

今回の取り組みで、Kaggle コンペの基本的な流れを確認しました。

特に、機械学習モデルに投入する前に、

- 欠損値を補完する
- 文字列データを数値化する
- train と test の列をそろえる

という前処理が重要であることを学びました。
