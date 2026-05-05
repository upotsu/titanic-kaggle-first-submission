import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Kaggle Notebook 上の Titanic データを読み込む。
    """
    train = pd.read_csv("/kaggle/input/titanic/train.csv")
    test = pd.read_csv("/kaggle/input/titanic/test.csv")
    return train, test


def extract_title(df):
    """
    Name から敬称を抽出する。
    例:
    Braund, Mr. Owen Harris -> Mr
    Cumings, Mrs. John Bradley -> Mrs
    """
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # 少数派の敬称をまとめる
    title_mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }
    df["Title"] = df["Title"].replace(title_mapping)

    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    return df


def create_features(train, test):
    """
    train と test を結合して、同じ前処理を行う。
    その後、再度 train/test に分割する。
    """
    train = train.copy()
    test = test.copy()

    train["is_train"] = 1
    test["is_train"] = 0

    combined = pd.concat([train, test], axis=0, sort=False)

    # 追加特徴量1: Name から Title を抽出
    combined = extract_title(combined)

    # 追加特徴量2: 家族人数
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1

    # 追加特徴量3: 一人乗船かどうか
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(int)

    # 追加特徴量4: Cabin 情報があるかどうか
    combined["HasCabin"] = combined["Cabin"].notnull().astype(int)

    # 欠損値補完
    combined["Age"] = combined["Age"].fillna(combined["Age"].median())
    combined["Fare"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["Embarked"] = combined["Embarked"].fillna(combined["Embarked"].mode()[0])

    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "HasCabin",
    ]

    X_all = combined[features]

    # カテゴリ変数を one-hot encoding
    X_all = pd.get_dummies(
        X_all,
        columns=["Sex", "Embarked", "Title"],
        drop_first=True
    )

    X_train = X_all[combined["is_train"] == 1]
    X_test = X_all[combined["is_train"] == 0]

    y = combined.loc[combined["is_train"] == 1, "Survived"].astype(int)

    return X_train, y, X_test


def compare_models(X, y):
    """
    複数モデルを Cross Validation で比較する。
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    print("Cross Validation Results")
    print("-" * 40)

    best_model_name = None
    best_score = 0
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="accuracy"
        )

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = model

    print("-" * 40)
    print(f"Best Model: {best_model_name}")
    print(f"Best CV Score: {best_score:.4f}")

    return best_model_name, best_model


def create_submission(model, X, y, X_test, test):
    """
    全学習データで再学習し、提出ファイルを作成する。
    """
    model.fit(X, y)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions.astype(int)
    })

    submission.to_csv("submission_v2.csv", index=False)
    print("submission_v2.csv を作成しました。")
    print(submission.head())


def main():
    train, test = load_data()

    print("train shape:", train.shape)
    print("test shape:", test.shape)

    X, y, X_test = create_features(train, test)

    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)
    print("features:", list(X.columns))

    best_model_name, best_model = compare_models(X, y)

    create_submission(best_model, X, y, X_test, test)


if __name__ == "__main__":
    main()