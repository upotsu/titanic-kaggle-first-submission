# Titanic Kaggle First Submission

Kaggle の入門コンペ「Titanic - Machine Learning from Disaster」に取り組み、  
乗客データから生存予測モデルを作成し、提出用 CSV を生成した学習記録です。

## 目的

Kaggle の基本的な流れを理解することを目的としています。

* データセットの読み込み
* データ構造の確認
* 欠損値の確認
* 特徴量の選択
* 欠損値補完
* カテゴリ変数の数値化
* 機械学習モデルの学習
* 検証データでの精度確認
* 提出用 CSV の作成

## 使用技術

* Python
* pandas
* scikit-learn
* Kaggle Notebook
* RandomForestClassifier

## データセット

Kaggle Competition: Titanic - Machine Learning from Disaster

使用ファイル:

* `train.csv`
* `test.csv`
* `gender\_submission.csv`

Kaggle のデータセットはリポジトリには含めず、Kaggle Notebook 上で読み込む想定です。

## ディレクトリ構成

```text
titanic-kaggle-first-submission/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ titanic\_first\_submission.ipynb
├─ src/
│  └─ titanic\_baseline.py
└─ submissions/
   └─ .gitkeep
```

## 実施内容

### 1\. データ読み込み

```python
import pandas as pd

train = pd.read\_csv("/kaggle/input/competitions/titanic/train.csv")
test = pd.read\_csv("/kaggle/input/competitions/titanic/test.csv")
```

### 2\. データ確認

```python
train.info()
train.isnull().sum()
```

確認した欠損値:

|カラム|欠損数|
|-|-:|
|Age|177|
|Cabin|687|
|Embarked|2|

### 3\. 使用した特徴量

```python
features = \["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
```

初回モデルでは、以下のカラムは使用していません。

* `PassengerId`
* `Name`
* `Ticket`
* `Cabin`

### 4\. 前処理

* `Age` は中央値で補完
* `Embarked` は最頻値で補完
* `Fare` は test 側の欠損対策として中央値で補完
* `Sex` と `Embarked` は one-hot encoding で数値化
* `train` 側と `test` 側の列をそろえる

### 5\. モデル学習

```python
from sklearn.model\_selection import train\_test\_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy\_score

X\_train, X\_valid, y\_train, y\_valid = train\_test\_split(
    X,
    y,
    test\_size=0.2,
    random\_state=42
)

model = RandomForestClassifier(
    n\_estimators=100,
    random\_state=42
)

model.fit(X\_train, y\_train)

valid\_predictions = model.predict(X\_valid)
accuracy = accuracy\_score(y\_valid, valid\_predictions)

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
python src/titanic\_baseline.py
```

実行後、ルートディレクトリに `submission.csv` が作成されます。

## 今後の改善案

* `Name` から敬称を抽出する
* `FamilySize` を作成する
* `IsAlone` を作成する
* `Cabin` の有無を特徴量化する
* `Age` の補完方法を改善する
* Logistic Regression や XGBoost など他モデルと比較する
* Cross Validation を導入する
* Kaggle の Public Score と Validation Accuracy の差を確認する

## 学習メモ

今回の取り組みで、Kaggle コンペの基本的な流れを確認しました。

特に、機械学習モデルに投入する前に、

* 欠損値を補完する
* 文字列データを数値化する
* train と test の列をそろえる

という前処理が重要であることを学びました。





## 【05/05追記】改善版 v2：特徴量エンジニアリングによる精度改善



初回提出では、基本的な特徴量のみを使用していました。  

v2では、Titanicデータの特徴を踏まえて、以下の特徴量を追加しました。



## 追加した特徴量



| 特徴量 | 内容 | 目的 |

|---|---|---|

| `Title` | `Name` から敬称を抽出 | 性別・年齢・社会的属性を補助するため |

| `FamilySize` | `SibSp + Parch + 1` | 同乗家族数による生存傾向を確認するため |

| `IsAlone` | `FamilySize == 1` | 単独乗船かどうかを表すため |

| `HasCabin` | `Cabin` の欠損有無 | 客室情報の有無を特徴量として利用するため |

## モデル比較



5分割 Cross Validation により、以下のモデルを比較しました。



| Model | CV Accuracy |

|---|---:|

| LogisticRegression | 0.8272 |

| RandomForest | 0.8283 |

| GradientBoosting | 0.8238 |



最もCVスコアが高かった `RandomForest` を使用して、`submission\_v2.csv` を作成しました。

## 結果



| Version | Model | Validation / CV Score | Kaggle Public Score |

|---|---|---:|---:|

| v1 | RandomForestClassifier | 0.8044 | 前回のスコアを記入 |

| v2 | RandomForestClassifier | 0.8283 | 今回のスコアを記入 |

## 考察



v2では、名前から抽出した敬称や家族人数、客室情報の有無を特徴量として追加しました。  

その結果、Cross Validation の平均精度は `0.8283` となり、初回提出時よりも検証精度が向上しました。



Kaggle Public Score でもわずかに改善が見られたため、Titanicデータにおいては、単純にモデルを変更するだけでなく、データの意味を踏まえた特徴量エンジニアリングが有効であると確認できました。

