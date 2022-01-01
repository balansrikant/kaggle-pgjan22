import optuna

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def objective_logreg(trial, X_in, y_in, X_val_in, y_val_in):
    """Optimize logistic regression model using optuna"""

    solver = trial.suggest_categorical(
        'solver',
        ['liblinear',
         'newton-cg',
         'lbfgs',
         'newton-cg',
         'sag',
         'saga'])
    C = trial.suggest_float("C", 0.01, 2.0)
    max_iter = trial.suggest_int("max_iter", 100, 10000, step=100)

    penalty = 'l2'

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        penalty=penalty)
    model.fit(X_in, y_in)
    preds = model.predict_proba(X_val_in)[:, 1]
    score = roc_auc_score(y_val_in, preds)

    return score


def objective_linearSVC(trial, X_in, y_in, X_val_in, y_val_in):
    """Optimize linear SVC model using optuna"""

    C = trial.suggest_float("C", 0.01, 2.0)
    max_iter = trial.suggest_int("max_iter", 1000, 10000, step=1000)

    model = LinearSVC(
        C=C,
        max_iter=max_iter,
        dual=False,
        random_state=5)
    clf = CalibratedClassifierCV(base_estimator=model, cv=5)
    clf.fit(X_in, y_in)

    preds = clf.predict_proba(X_val_in)[:, 1]
    score = roc_auc_score(y_val_in, preds)

    return score


def objective_lightgbm(trial, X_in, y_in, X_val_in, y_val_in):
    """optuna objective function for lightgbm"""

    num_leaves = trial.suggest_int("num_leaves", 11, 101, step=10)
    max_depth = trial.suggest_int("max_depth", 2, 10, step=1)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0)

    model = LGBMClassifier(
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='binary',
        random_state=5)
    model.fit(X_in, y_in)
    preds = model.predict_proba(X_val_in)[:, 1]
    score = roc_auc_score(y_val_in, preds)

    return score


def optimize_linear_svc(X_in, y_in, X_val_in, y_val_in, n_trials):
    """optimize linear svc using optuna"""
    study = optuna.create_study(direction='maximize', study_name="LinearSVC")
    func = lambda trial: objective_linearSVC(
        trial,
        X_in,
        y_in,
        X_val_in,
        y_val_in)
    study.optimize(func, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best score:', study.best_value)

    return study.best_trial.params


def optimize_logreg(X_in, y_in, X_val_in, y_val_in, n_trials):
    """optimize logistic regression using optuna"""
    study = optuna.create_study(
        direction='maximize',
        study_name="Logistic regression")
    func = lambda trial: objective_logreg(
        trial,
        X_in,
        y_in,
        X_val_in,
        y_val_in)
    study.optimize(func, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best score:', study.best_value)

    return study.best_trial.params


def optimize_lightgbm(X_in, y_in, X_val_in, y_val_in, n_trials):
    """optimize lightgbm using optuna"""
    study = optuna.create_study(direction='maximize', study_name="LightGBM")
    func = lambda trial: objective_lightgbm(trial)
    study.optimize(func, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best score:', study.best_value)

    return study.best_trial.params
