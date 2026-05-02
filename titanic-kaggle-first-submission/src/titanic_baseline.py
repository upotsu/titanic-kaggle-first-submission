import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


TRAIN_PATH = "/kaggle/input/competitions/titanic/train.csv"
TEST_PATH = "/kaggle/input/competitions/titanic/test.csv"


def load_data():
    """Load Titanic train and test data from Kaggle input path."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def preprocess(train, test):
    """Select features, fill missing values, and encode categorical variables."""
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    X = train[features].copy()
    y = train["Survived"]
    X_test = test[features].copy()

    # Fill missing values
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X_test["Age"] = X_test["Age"].fillna(X["Age"].median())

    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])
    X_test["Embarked"] = X_test["Embarked"].fillna(X["Embarked"].mode()[0])

    X_test["Fare"] = X_test["Fare"].fillna(X["Fare"].median())

    # Convert categorical columns to numeric columns
    X = pd.get_dummies(X, columns=["Sex", "Embarked"])
    X_test = pd.get_dummies(X_test, columns=["Sex", "Embarked"])

    # Align train and test columns
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    return X, y, X_test


def train_and_validate(X, y):
    """Train RandomForest model and evaluate validation accuracy."""
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

    print(f"Validation Accuracy: {accuracy}")

    return model


def create_submission(model, X, y, X_test, test):
    """Retrain model on full train data and create submission.csv."""
    model.fit(X, y)

    test_predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_predictions
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv created.")


def main():
    train, test = load_data()
    X, y, X_test = preprocess(train, test)

    model = train_and_validate(X, y)

    create_submission(model, X, y, X_test, test)


if __name__ == "__main__":
    main()
