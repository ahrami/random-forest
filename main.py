import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("student_performance.csv")
df_class = df.drop(columns=["Ethnicity", "StudentID", "GPA"])
df_regr = df.drop(columns=["Ethnicity", "StudentID", "GradeClass"])


def rf_classification_report(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    print("Score", rf.score(X_test, y_test))
    print(classification_report(y_test, y_pred))

    features = pd.DataFrame(rf.feature_importances_, index=X_test.columns)
    print(features.head(15))


def rf_regression_report(rf, X_test, y_test):
    y_pred = rf.predict(X_test)
    print("Score", rf.score(X_test, y_test))
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))
    print("R2", r2_score(y_test, y_pred))

    features = pd.DataFrame(rf.feature_importances_, index=X_test.columns)
    print(features.head(15))


def classification():
    X = df_class.drop(["GradeClass"], axis=1)
    y = df_class["GradeClass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    rf_classification_report(rf, X_test, y_test)


def classification_better():
    X = df_class.drop(["GradeClass"], axis=1)
    y = df_class["GradeClass"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion="entropy",
        min_samples_split=10,
        max_depth=14,
    )
    rf.fit(X_train, y_train)

    rf_classification_report(rf, X_test, y_test)


def regression():
    X = df_regr.drop(["GPA"], axis=1)
    y = df_regr["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    rf_regression_report(rf, X_test, y_test)


def regression_better():
    X = df_regr.drop(["GPA"], axis=1)
    y = df_regr["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    rf = RandomForestRegressor(
        n_estimators=1000,
        min_samples_split=2,
        max_depth=30,
        min_samples_leaf=1,
    )
    rf.fit(X_train, y_train)

    rf_regression_report(rf, X_test, y_test)


def regression_cv():
    X = df_regr.drop(["GPA"], axis=1)
    y = df_regr["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [20, 30],
        "min_samples_split": [2, 5],
        # "min_samples_leaf": [1, 3],
    }

    rf = RandomForestRegressor()

    rf_cv = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error"
    )

    rf_cv.fit(X_train, y_train)

    print("Best Estimator", rf_cv.best_params_)

    y_pred = rf_cv.predict(X_test)
    print("Score", rf_cv.score(X_test, y_test))
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))
    print("R2", r2_score(y_test, y_pred))
