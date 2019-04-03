from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics.scorer import make_scorer
from sklearn.base import clone
import sys
from nafld_config import SEED
import sklearn
import logging
from utils import generate_seeds
import numpy as np
from scipy import stats


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG) 
logging.info(f"sciki-learn package version: {sklearn.__version__}")


def parameter_optimization(*, model=None, parameters_space=None, dx=None, dy=None, mode='r', scoring='accuracy',
                           cv=5, n_jobs=1, n_iter=10, seed=None, fit_params=None):
    # scoring options: https://scikit-learn.org/stable/modules/model_evaluation.html
    # commonly used scoring method: roc_auc, f1 (for binary), we also provide pr_auc
    # model has two options, r==random; g==grid
    if scoring == "pr_auc":
        scoring = make_scorer(average_precision_score, needs_proba=True, needs_threshold=True)

    cv_model = None

    if mode == 'r':
        cv_model = RandomizedSearchCV(model, parameters_space, scoring=scoring, n_jobs=n_jobs, cv=StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=seed), verbose=1, iid=True, n_iter=n_iter, random_state=seed)
    elif mode == 'g':
        cv_model = GridSearchCV(model, parameters_space, scoring=scoring, n_jobs=n_jobs, cv=StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=seed), verbose=1, iid=True)
    else:
        raise RuntimeError('mode must be r or g for RandomizedSearchCV or GridSearchCV')

    if fit_params:
        cv_model.fit(dx, dy, groups=None, **fit_params)
    else:
        cv_model.fit(dx, dy)
    logging.info(f"best model:\n {cv_model.best_estimator_}")
    logging.info(f"parameters tuning best results (averaged): {cv_model.best_score_}")

    return cv_model.best_estimator_


def get_tpr_fpr_youden(model, dx, dy, seed, n=5, n_jobs=1):
    # #the code below yield the same results as current used code
    # y_preds = []
    # y_trues = []
    # sk = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    # for train_idx, test_idx in sk.split(dx, dy):
    #     train_x, train_y = dx[train_idx], dy[train_idx]
    #     test_x, test_y = dx[test_idx], dy[test_idx]
    #     sub_model = clone(model)
    #     sub_model.fit(train_x, train_y)
    #     sub_preds = sub_model.predict_proba(test_x)
    #     y_preds.extend(list(map(lambda x: x[-1], sub_preds)))
    #     y_trues.extend(test_y)
    # fprs, tprs, th = roc_curve(y_trues, y_preds)

    y_preds = cross_val_predict(model, dx, dy, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=seed),
                          n_jobs=n_jobs, method="predict_proba")
    fprs, tprs, th = roc_curve(dy, y_preds[:, -1])
    idx = np.argmax(np.abs(tprs - fprs))
    fpr, tpr, J = fprs[idx], tprs[idx], th[idx]
    logging.info(f"5-fold CV final results on all prediction (not averaged): {roc_auc_score(dy, y_preds[:, -1])}")
    return auc(fprs, tprs), tpr, fpr, J, fprs, tprs


def m_x_n_cv_performance(m=20, n=5, model=None, dx=None, dy=None, scoring='accuracy', n_jobs=1):
    # this will run a m x n fold cross validation on the whole dataset for performance evaluation
    # default is 20 x 5 folds => 5-fold cv run 20 times
    seeds = generate_seeds(m)
    logging.info(f"seeds: {seeds}")
    results = []

    for seed in seeds:
        scores = cross_val_score(model, dx, dy, scoring=scoring, cv=StratifiedKFold(n_splits=n, shuffle=True,
                                                                                    random_state=seed), n_jobs=n_jobs)
        results.extend(scores)

    mu, sigma = np.mean(results), np.std(results)
    ci95 = stats.t.interval(0.95, len(results)-1, mu, stats.sem(results))

    return results, mu, sigma, ci95


# def m_x_n_cv_tpr_fpr(m=20, n=5, model=None, dx=None, dy=None, scoring='accuracy', n_jobs=1):
#     seeds = generate_seeds(m)
#     logging.info(f"seeds: {seeds}")
#     trues = []
#     preds = []
#
#     for seed in seeds:
#         pred_probs = cross_val_predict(model, dx, dy, cv=StratifiedKFold(n_splits=n, shuffle=True, random_state=seed),
#                        method='predict_proba')
#         preds.extend(pred_probs)
#
#     y_pred = list(map(lambda x: x[-1], preds))
