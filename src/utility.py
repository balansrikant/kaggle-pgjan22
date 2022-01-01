import copy
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_datasets(path: str, scale: bool, debug: bool):
    """Import datasets from path. Expect csvs called train.csv and test.csv

    Arguments:
    :path - path containing csvs
    :scale - run standard scaler
    :debug - run in debug mode

    Returns:
    :X - dataframe (train) minus target
    :y - series (target values for train)
    :df_test - dataframe (test)
    """

    if debug:
        df_train = pd.read_csv(path + 'train.csv', nrows=1000)
        df_test = pd.read_csv(path + 'test.csv', nrows=1000)
    else:
        df_train = pd.read_csv(path + 'train.csv')
        df_test = pd.read_csv(path + 'test.csv')

    ids = df_test.Id
    df_train.drop('Id', axis=1, inplace=True)
    df_test.drop('Id', axis=1, inplace=True)

    original_features = df_test.columns

    X = df_train[original_features]
    y = df_train['Cover_Type']

    if scale:
        std_scaler = StandardScaler()
        X_norm = pd.DataFrame(std_scaler.fit_transform(X))
        X_norm.columns = original_features
        df_test_norm = pd.DataFrame(std_scaler.transform(df_test))
        df_test_norm.columns = original_features
    else:
        X_norm = X
        df_test_norm = df_test

    return X_norm, y, df_test_norm, ids


def get_models():
    """Return list of models for initial analysis

    Returns:
    :models - list of dicts(name, model)
    """
    models = [
        {'name': 'lr', 'model': LogisticRegression(random_state=5)},
        # {'name': 'lsvc', 'model': LinearSVC(dual=False, random_state=5)},
        # {'name': 'lgbm', 'model': LGBMClassifier(random_state=5)},
        # {'name': 'bayes', 'model': GaussianNB()},
    ]
    return models


def evaluate_model(
        model,
        X: pd.DataFrame,
        y: pd.Series) -> list:
    """Return list of scores for a model

    Arguments:
    :model - model to be evaluated
    :X - dataframe (train) minus target
    :y - series (target values for train)

    Returns:
    scores - list of scores for model
    """

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    scores = cross_val_score(
        model,
        X,
        y,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        error_score='raise')

    return scores


def evaluate_model_val_set(
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series) -> float:
    """Return scores for a model on validation set

    Arguments:
    :model - model to be evaluated
    :X_train - training dataframe minus target
    :y_train - training series (target values for training set)
    :X_val - validation dataframe minus target
    :y_val - validation series (target values for validation set)

    Returns:
    score - score for model
    """
    if model.__class__.__name__ == 'LinearSVC':
        clf = CalibratedClassifierCV(base_estimator=model, cv=5)
    else:
        clf = model
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    score = accuracy_score(y_val, preds)

    return score


def get_feature_importances(
        X_in: pd.DataFrame,
        y_in: pd.Series,
        k: int) -> pd.DataFrame:
    """Return feature importances of features as to the target prediction

    Arguments:
    :X_in - dataframe (train) minus target
    :y_in - series (target values for train)
    :model_type - 'regression' or 'classification'
    :k - number of folds

    Returns:
    :featureScores - dataframe with abs correlation value sorted in asc
    """
    bestfeatures = SelectKBest(score_func=f_classif, k=k)

    fit = bestfeatures.fit(X_in, y_in)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_in.columns)

    # Concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    featureScores['Abs_score'] = abs(featureScores['Score'])
    featureScores.sort_values(by='Score', axis=0, ascending=True, inplace=True)
    featureScores.reset_index(drop=True, inplace=True)

    plt.bar(featureScores['Specs'], featureScores['Abs_score'])
    plt.title('Feature Importances')
    plt.show()

    return featureScores


def get_kmeans_labels(
        X_in: pd.DataFrame,
        features: list,
        n_clusters: int) -> list:
    """Return kmeans labels for a dataframe

    Arguments:
    :X_in - dataframe (train) minus target
    :features - list of important features
    :n_clusters - number of kmeans clusters

    Returns:
    X_temp - dataframe (train) minus target plus kmeans labels
    """
    X_temp = copy.deepcopy(X_in)
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    kmeans.fit(X_temp[features])
    X_temp['cluster'] = kmeans.predict(X_temp[features])

    return X_temp


def get_kmeans_dist_ratios(
        X_in: pd.DataFrame,
        X_val_in: pd.DataFrame,
        X_test_in: pd.DataFrame,
        features: list,
        n_clusters: int) -> list:
    """Return kmeans labels for a dataframe

    Arguments:
    :X_in - dataframe (train) minus target
    :X_val_in - dataframe (val) minus target
    :X_test_in - dataframe (test) minus target
    :features - list of important features
    :n_clusters - number of kmeans clusters

    Returns:
    :X_temp - dataframe (train) minus target plus kmeans dist ratios
    :X_temp_val - dataframe (val) minus target plus kmeans dist ratios
    :X_temp_test - dataframe (test) minus target plus kmeans dist ratios
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=3)
    X_temp = copy.deepcopy(X_in)
    X_temp_val = copy.deepcopy(X_val_in)
    X_temp_test = copy.deepcopy(X_test_in)

    kmeans.fit(X_temp[features])
    cluster_cols = [f"cluster{i+1}" for i in range(n_clusters)]

    cd = kmeans.transform(X_temp[features])
    cd_val = kmeans.transform(X_temp_val[features])
    cd_test = kmeans.transform(X_temp_test[features])

    X_temp_cd = pd.DataFrame(
        cd,
        columns=cluster_cols,
        index=X_temp.index)
    X_temp_val_cd = pd.DataFrame(
        cd_val,
        columns=cluster_cols,
        index=X_temp_val.index)
    X_temp_test_cd = pd.DataFrame(
        cd_test,
        columns=cluster_cols,
        index=X_temp_test.index)

    # calculate cluster distances (cd)
    new_cols = []
    for i in cluster_cols:
        for j in cluster_cols:
            if i != j:
                new_col_name = i + '_' + j
                X_temp_cd[new_col_name] = X_temp_cd[i] / X_temp_cd[j]
                X_temp_val_cd[new_col_name] = \
                    X_temp_val_cd[i] / X_temp_val_cd[j]
                X_temp_test_cd[new_col_name] = \
                    X_temp_test_cd[i] / X_temp_test_cd[j]
                new_cols.append(new_col_name)

    X_temp = X_temp.join(X_temp_cd[new_cols])
    X_temp_val = X_temp_val.join(X_temp_val_cd[new_cols])
    X_temp_test = X_temp_test.join(X_temp_test_cd[new_cols])

    return X_temp, X_temp_val, X_temp_test


def generate_meta_features_model(model, X_in, y_in, cv):
    """Generate meta features for single base classifier model.
     to be used later for stacking

    Arguments:
    :model - model to evaluate
    :X_in - dataframe with features minus target
    :y_in - target series
    :cv - cross-validation iterator
    """

    # Initialize

    # Assuming that train data contains all classes
    n_classes = len(np.unique(y_in))
    meta_features = np.zeros((X_in.shape[0], n_classes))
    n_splits = cv.get_n_splits(X_in, y_in)

    # Loop over folds
    print("Starting hold out prediction with {} splits for {}."
          .format(n_splits, model.__class__.__name__))
    for train_idx, hold_out_idx in cv.split(X_in, y_in):

        # Split data
        X_in_train = X_in.iloc[train_idx]
        y_in_train = y_in.iloc[train_idx]
        X_in_hold_out = X_in.iloc[hold_out_idx]

        # Fit estimator to K-1 parts and predict on hold out part
        est = copy.deepcopy(model)
        est.fit(X_in_train, y_in_train)
        y_in_hold_out_pred = est.predict_proba(X_in_hold_out)

        # Fill in meta features
        meta_features[hold_out_idx] = y_in_hold_out_pred

    return meta_features


def get_stack_df(models, X_input, X_input_km, y_in):
    # Loop over classifier to produce meta features
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    meta_train = []
    for model in models:
        name = model['name']
        if name == 'lr':
            X_in = X_input
        elif name == 'lsvc':
            X_in = X_input
        else:
            X_in = X_input_km

        # Create hold out predictions for a classifier
        if model['model'].__class__.__name__ == 'LinearSVC':
            clf = CalibratedClassifierCV(base_estimator=model['model'], cv=5)
        else:
            clf = model['model']
        meta_train_model = generate_meta_features_model(clf, X_in, y_in, cv)

        # Remove extracolumn - 0th col = 1st col in a two class dataset
        meta_train_model = np.delete(meta_train_model, 0, axis=1).ravel()
        print(pd.DataFrame(meta_train_model).head())

        # Gather meta training data
        meta_train.append(meta_train_model)

        meta_train = np.array(meta_train).T
        df_meta_train = pd.DataFrame(meta_train)

        # Optional (Add original features to meta)
        df_meta_train = pd.DataFrame(np.concatenate(
            (df_meta_train, X_in), axis=1))

        return df_meta_train


def get_stack_df_val(
        models,
        stack_model,
        X_input,
        X_input_km,
        y_in,
        X_test_input,
        X_test_km_input,
        features,
        ids):

    meta_test = []
    for model in models:
        name = model['name']
    if name == 'lr':
        X_in = X_input
        X_test_in = X_test_input
    elif name == 'lsvc':
        X_in = X_input
        X_test_in = X_test_input
    else:
        X_in = X_input_km
        X_test_in = X_test_km_input

    stack_model.fit(X_in, y_in)
    meta_test_model = stack_model.predict_proba(X_test_in)

    # Remove redundant column - 0th col = 1st col in a two class dataset
    meta_test_model = np.delete(meta_test_model, 0, axis=1).ravel()

    # Gather meta training data
    meta_test.append(meta_test_model)

    meta_test = np.array(meta_test).T
    df_meta_test = pd.DataFrame(meta_test)

    # Optional (Add original features to meta)
    df_meta_test = pd.DataFrame(np.concatenate((
        df_meta_test,
        X_test_in), axis=1))

    return df_meta_test


def get_baseline_scores(models, X_in, y_in):
    """evaluate baseline cross val scores of the models"""
    results, names = list(), list()
    for model in models:
        name = model['name']
        scores = evaluate_model(model['model'], X_in, y_in)
        model['init_scores'] = np.mean(scores)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.title('Baseline Scores')
    plt.show()


def get_baseline_scores_validation(models, X_in, y_in, X_val_in, y_val_in):
    """evaluate baseline scores of the models on validation set"""
    for model in models:
        name = model['name']
        score = evaluate_model_val_set(
            model['model'],
            X_in,
            y_in,
            X_val_in,
            y_val_in)
        model['init_scores_val'] = score
        print('>%s %.3f' % (name, score))


def get_kmeans_scores(featureScores, X_in, y_in, X_val_in, X_test_in, models):
    """evaluate cross val scores by adding kmeans cluster distance ratios"""
    results, names = list(), list()
    important_features = list(featureScores.sort_values(
        by='Abs_score',
        ascending=False).head(15)['Specs'])
    X_train_km, X_val_km, X_test_km = get_kmeans_dist_ratios(
        X_in,
        X_val_in,
        X_test_in,
        important_features,
        10)
    for model in models:
        name = model['name']
        if name == 'lr':
            pass
        elif name == 'lsvc':
            pass
        else:
            scores = evaluate_model(model['model'], X_train_km, y_in)
            model['kmeans_dist_rat_scores'] = np.mean(scores)
            results.append(scores)
            names.append(name)
            print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.title('KMeans Distance Ratio Scores')
    plt.show()


def get_kmeans_scores_validation(
        featureScores,
        X_in,
        y_in,
        X_val_in,
        y_val_in,
        X_test_in,
        models):
    """evaluate scores by adding kmeans cluster distance ratios on val set"""
    important_features = list(featureScores.sort_values(
        by='Abs_score',
        ascending=False).head(15)['Specs'])
    X_train_km, X_val_km, X_test_km = get_kmeans_dist_ratios(
        X_in,
        X_val_in,
        X_test_in,
        important_features,
        10)
    for model in models:
        name = model['name']
        if name == 'lr':
            pass
        elif name == 'lsvc':
            pass
        else:
            score = evaluate_model_val_set(
                model['model'],
                X_train_km,
                y_in,
                X_val_km,
                y_val_in)
            model['kmeans_dist_rat_scores_val'] = score
            print('>%s %.3f' % (name, score))


def get_stacking_scores(
        featureScores,
        X_in,
        y_in,
        X_val_in,
        X_test_in,
        models,
        stack_model):
    """evaluate cross val scores with stacking"""
    # get km enhanced df
    important_features = list(featureScores.sort_values(
        by='Abs_score',
        ascending=False).head(15)['Specs'])
    X_train_km, X_val_km, X_test_km = get_kmeans_dist_ratios(
        X_in,
        X_val_in,
        X_test_in,
        important_features,
        10)

    # get meta df
    df_meta_train = get_stack_df(models, X_in, X_train_km, y_in)

    # cross val scores from stacking
    results = list()
    name = [model['name'] + '_' for model in models]
    scores = evaluate_model(stack_model, df_meta_train, y_in)
    results.append(scores)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


def get_binning_scores(baseline_score, X_in, y_in, model_in, improvement):
    """check cross val score for improvement in score by binning column"""
    improved_cols = []
    for col in X_in.columns:
        print(col)
        if model_in.__class__.__name__ == 'LinearSVC':
            clf = CalibratedClassifierCV(base_estimator=model_in, cv=10)
        else:
            clf = model_in

        X_new = copy.deepcopy(X_in)
        new_col = col + '_bin'
        X_new[new_col], bins = pd.qcut(
            X_in[col],
            q=1000,
            retbins=True,
            labels=False)

        scores = evaluate_model(clf, X_new, y_in)
        new_score = scores.mean()
        if new_score >= baseline_score + 0.00001:
            new_col = {'col': col, 'score': new_score}
            improved_cols.append(new_col)
    return improved_cols


def get_binning_scores_val(
        baseline_score,
        X_in,
        y_in,
        X_val_in,
        y_val_in,
        model_in,
        improvement):
    """check score for improvement in score by binning column"""
    improved_cols = []

    for col in X_in.columns:
        print(col)
        if model_in.__class__.__name__ == 'LinearSVC':
            clf = CalibratedClassifierCV(base_estimator=model_in, cv=10)
        else:
            clf = model_in

        X_new = copy.deepcopy(X_in)
        X_val_new = copy.deepcopy(X_val_in)
        new_col = col + '_bin'

        X_new[new_col], bins = pd.qcut(
            X_in[col],
            q=1000,
            retbins=True,
            labels=False)
        X_val_new[new_col] = pd.cut(
            X_val_new[col],
            bins=bins,
            labels=False,
            include_lowest=True)
        X_val_new[new_col].fillna(X_val_new[new_col].mode()[0], inplace=True)

        score = evaluate_model_val_set(clf, X_new, y_in, X_val_new, y_val_in)
        if score >= baseline_score + improvement:
            new_col = {'col': col, 'score': score}
            improved_cols.append(new_col)
    return improved_cols


def split_dataset(df_train_in, df_test_in, target):
    original_features = df_test_in.columns
    X = df_train_in[original_features]
    y = df_train_in[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        train_size=0.8, 
        stratify=y)

    return X_train, X_val, y_train, y_val, original_features


def make_final_pred_single(model_in, X_in, y_in, X_test_in, ids):
    """make final test predictions using a single model"""
    if model_in.__class__.__name__ == 'LinearSVC':
        clf = CalibratedClassifierCV(base_estimator=model_in, cv=10)
    else:
        clf = model_in

    clf.fit(X_in, y_in)
    preds = clf.predict_proba(X_test_in)[:, 1]

    output = pd.DataFrame({'id': ids, 'target': preds})
    output.to_csv('submission.csv', index=False)
