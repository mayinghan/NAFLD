from machine_learning import parameter_optimization, m_x_n_cv_performance, get_tpr_fpr_youden
from nafld_config import SEED
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)


def auc_expr(clf, tuned_parameters, dx, dy, score_method):
    # optimize model
    # set n_jobs to -1 in XGBClassifier, RandomForestClassifier
    nj = -1
    fit_params = None
    if isinstance(clf, XGBClassifier):
        # nj = 1
        clf.set_params(n_jobs=-1, verbosity=0)
        fit_params = {"verbose": False}
    elif isinstance(clf, RandomForestClassifier):
        clf.set_params(n_jobs=-1)

    # in this experiment, we only randomized on 50% parameters
    p_num = 1
    for k, v in tuned_parameters.items():
        p_num *= len(v)
    ni = int(0.25*p_num)

    logging.info(f"total number of parameter combinations searched: {ni}")

    search_mode = "r"
    optimized_model = parameter_optimization(model=clf, parameters_space=tuned_parameters, dx=dx, dy=dy,
                                            mode=search_mode, scoring=score_method, cv=5, n_jobs=nj, n_iter=ni,
                                             seed=SEED, fit_params=fit_params)

    # get tpr, fpr, and J-index
    auc_roc, tpr, fpr, J, fprs, tprs = get_tpr_fpr_youden(optimized_model, dx, dy, SEED, 5, nj)

    # run 20 x 5 cv to evaluate performance
    pred_scores, mu, sigma, ci95 = m_x_n_cv_performance(m=20, n=5, model=optimized_model, dx=dx, dy=dy,
                                                        scoring=score_method, n_jobs=nj)

    d = {'tpr': tpr, 'fpr': fpr, 'youden_index': J.item(), "cv_mean": mu, "cv_std": sigma, "95ci": ci95,
         'performance': auc_roc, 'eval_metrics': score_method, 'cv_scores': pred_scores, "fpr_list": fprs.tolist(),
         "tpr_list": tprs.tolist()}

    return optimized_model, auc_roc, d


def two_results_t_test(list1, list2, eq_var=False):
    return stats.ttest_ind(list1, list2, equal_var=eq_var)
