import re
import warnings
from os.path import join

import catboost as ctb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import xgboost as xgb
from catboost import Pool
from graphviz import Source
from scipy.spatial.distance import euclidean
from six import StringIO
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GroupKFold
from sklearn.tree import export_graphviz
from tqdm import tqdm

from ..gentools.advtools import cut_feas
from ..plot import feature_plot as fplt


def evaluate_performance(all_target, predicted, to_plot=True, suptitle='', bins=10, save_path=None):
    """
    Evaluate performance of a score with KS/AUC, score distribution/average score, etc.

    :param all_target:  real target
    :param predicted:   predicted probability
    :param to_plot:  plot performance image if True
    :param suptitle:  suptitle of performance image
    :param bins:  number of bins for binning
    :param save_path: abspath to save performance image
    :return:

    Examples:
        (1) ks, roc_auc, fig, show_list = mdl.evaluate_performance(x_train[target], x_train['lr_pred'],
                         to_plot=True, suptitle='x_train performance', bins=10, save_path=None)
        (2) ks, roc_auc, fig, show_list = mdl.evaluate_performance(x_train[target], x_train['lr_pred'],
                         to_plot=True, suptitle='x_train performance', bins=[-np.inf, 0.02, 0.03, 0.05, 0.06, 0.1, 0.2, np.inf],
                         save_path='E:/huarui/')
        (3) ks, roc_auc, show_list = mdl.evaluate_performance(x_train[target], x_train['lr_pred'],
                         to_plot=False, suptitle='', bins=20, save_path=None)
    """

    # Change predict
    idx = predicted > 1
    if idx.sum() > 0:
        predicted = predicted / predicted.max()

    # Model performance
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = round(auc(fpr, tpr), 4)
    ks = max(tpr - fpr)
    try:
        i_max = np.argwhere(tpr - fpr == ks)[0]
    except IndexError:
        # IndexError: index 0 is out of bounds for axis 0 with size 0
        warnings.warn(f'Failed to evaluate performance when ks={ks}, roc_auc={roc_auc} with IndexError')
        return ks, roc_auc

    event_cnt = int(sum(all_target))
    nonevent_cnt = all_target.shape[0] - event_cnt
    event_rate = event_cnt / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1 - event_rate)
    i_min = np.argmin(abs(cum_total - event_rate))
    # Find thresh closest to perfect point [0, 1]
    dist = np.apply_along_axis(lambda x: euclidean([0, 1], x), 1, np.vstack([fpr, tpr]).T)
    idx_min = np.nonzero(dist == min(dist))[0][0]
    best_point = [fpr[idx_min], tpr[idx_min]]
    best_cutoff = thresholds[idx_min]

    # Info
    show_list = []
    show_list.append('KS=' + str(round(ks, 3)) + ', AUC=' + str(round(roc_auc, 3)) + ', N=' + str(predicted.shape[0]))
    show_list.append(f'Point closest to [0, 1] is FPR={best_point[0]:.3f}/TPR={best_point[1]:.3f},'
                     f' with threshold={best_cutoff:.3f}')
    show_list.append('At threshold={:.3f}, TPR={:.3f} ({:d} out of {:d}), FPR={:.3f} ({:d} out of {:d})'
                     .format(event_rate,
                             tpr[i_min], int(round(tpr[i_min] * event_cnt)), event_cnt,
                             fpr[i_min], int(round(fpr[i_min] * nonevent_cnt)), nonevent_cnt))

    # Score average by percentile
    df = pd.DataFrame({'target': all_target, 'pred': predicted})
    cut = cut_feas(df, 'pred', 'target', cut_params=bins)
    df['bin'] = pd.cut(df['pred'], cut, labels=range(len(cut) - 1))
    stats = df.groupby('bin').agg({'target': ['count', 'mean'], 'pred': 'mean'}).round(6)
    show_list.append('Ave_target: ' + str(stats[('target', 'mean')].tolist()))
    show_list.append('Ave_predicted: ' + str(stats[('pred', 'mean')].tolist()))

    if to_plot:
        # Start plotting
        fig = plt.figure(figsize=(20, 15))

        # KS plot
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        plt.title(f'KS={str(round(ks, 3))} AUC={str(round(roc_auc, 3))}\n  ROC    curve', fontsize=25)
        plt.plot([fpr[i_max], fpr[i_max]], [fpr[i_max], tpr[i_max]], linewidth=4, color='r')
        plt.plot([fpr[i_min]], [tpr[i_min]], 'k.', markersize=10)
        plt.plot([best_point[0]], [best_point[1]], 'rx', markersize=10)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False positive', fontsize=25)
        plt.ylabel('True positive', fontsize=25)

        # Score distribution
        plt.subplot(2, 2, 2)
        plt.hist(predicted, bins=20)
        plt.axvline(x=np.mean(predicted), linestyle='--')
        plt.axvline(x=np.mean(all_target), linestyle='--', color='g')
        plt.title('N=' + str(all_target.shape[0]) + ' True=' + str(round(np.mean(all_target), 3)) + ' Pred=' + str(
            round(np.mean(predicted), 3)), fontsize=25)
        plt.xlabel('Target rate', fontsize=25)
        plt.ylabel('Count', fontsize=25)

        # ks pic
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, tpr, label='tpr', markersize=5)
        plt.plot(thresholds, fpr, label='fpr', markersize=5)
        plt.plot((thresholds[i_max], thresholds[i_max]), (0, 1),
                 label='threshold is {}'.format(round(thresholds[i_max][0], 3)),
                 color='r', markersize=5)
        plt.legend(loc='upper right')
        plt.title('KS    curve', fontsize=25)
        plt.xlim(0, round(np.quantile(thresholds, 0.99), 3))
        plt.ylim(0, 1)
        plt.xlabel('thresholds', fontsize=25)
        plt.ylabel('percentage', fontsize=25)

        # Bin performance
        ax1 = plt.subplot(2, 2, 4)
        ax1.bar(stats.index.tolist(), stats[('target', 'count')].tolist(), label='count', alpha=0.2)
        ax2 = ax1.twinx()
        ax2.plot(stats.index.tolist(), stats[('pred', 'mean')].tolist(), 'b.-', label='Prediction', markersize=5)
        ax2.plot(stats.index.tolist(), stats[('target', 'mean')].tolist(), 'r.-', label='Truth', markersize=5)
        plt.title('cut bins', fontsize=25)

        plt.legend(loc='upper left')
        plt.xlabel('Percentile', fontsize=25)
        plt.ylabel('Target rate', fontsize=25)

        if suptitle:
            plt.suptitle(suptitle, fontsize=28)
            plt.subplots_adjust(top=0.85)
        if save_path is not None:
            # Save image
            plt.savefig(join(save_path, f'{suptitle}.png'), facecolor='w', bbox_inches='tight')

        plt.close(fig)

        return ks, roc_auc, fig, show_list

    return ks, roc_auc, show_list


def get_feature_names(model):
    """
    get the name of features in model

    :param model: model
    :return:

    Examples:
        feas = get_feature_names(model)
    """

    if re.search('statsmodels', str(model)):
        return list(model.params[:-1].index)

    elif re.search('sklearn.tree|sklearn.ensemble', str(type(model))):
        return model.feature_names_in_

    elif re.search('catboost', str(model)):
        return model.feature_names_

    elif re.search('xgboost', str(model)):
        return model.feature_names

    elif re.search('lightgbm', str(model)):
        return model.feature_name()

    else:
        raise ValueError(
            'only support LR/DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting/Xgboost/Lightgbm/Catboost')


def get_pred_cutoff(data, score, target, cut_params, ispred=False):
    """
    Show prediction cutoff performance

    :param data: data
    :param score: prediction
    :param target: target
    :param cut_params: cut_params
    :param ispred: whether to calc prediction mean
    :return:

    Examples:
        st_show = get_pred_cutoff(x_train, score='lr_pred', target=target, cut_params=10, ispred=True)[1]
    """

    bins = cut_feas(data, score, target, cut_params=cut_params)
    tmp = data[[score] + [target]]
    if cut_params is not None:
        tmp[score] = pd.cut(tmp[score], bins)

    st = fplt.PlotUtils()._get_state(tmp, score, target)
    st.rename(columns={'count': '总样本数', target: '坏样本占比'}, inplace=True)

    st['坏样本数'] = (st['总样本数'] * st['坏样本占比']).astype(int)
    st['好样本数'] = (st['总样本数'] - st['坏样本数']).astype(int)
    st['好样本占比'] = st['好样本数'] / st['总样本数']
    st = st[['总样本数', '坏样本数', '好样本数', '坏样本占比', '好样本占比']]

    # get 累计
    num_ = list(st['坏样本数'])
    st['累计坏样本数'] = [sum(num_[:i + 1]) for i in range(len(num_))]
    num_ = list(st['好样本数'])
    st['累计好样本数'] = [sum(num_[:i + 1]) for i in range(len(num_))]
    st['累计坏样本占比'] = st['累计坏样本数'] / st['累计坏样本数'].tail(1).tolist()[0]
    st['累计好样本占比'] = st['累计好样本数'] / st['累计好样本数'].tail(1).tolist()[0]
    st['ks'] = abs(st['累计好样本占比'] - st['累计坏样本占比'])
    st['lift'] = st['坏样本占比'] / data[target].mean()

    if ispred is True:
        st['预测坏样本概率'] = [data[(bins[i] < data[score]) & (data[score] <= bins[i + 1])][score].mean() for i in
                         range(len(bins) - 1)]

    st_show = st.copy()
    for c in [c for c in st_show if re.search('占比|ks', c)]:
        st_show[c] = st_show[c].apply(lambda x: f'{round(100 * x, 2)}%')

    return st, st_show, bins


class ModelImportance:

    @classmethod
    def lr_importance(cls, model, importance_type='all'):
        """
        Return the importance of features into lr model

        :param model: lr model
        :param importance_type: support str or list like 'all'/'pval'/['zscore','pval','occur_time']
        :return:

        Examples:
            (1) feas_imp = lr_importance(lm, importance_type='all')
            (2) feas_imp = lr_importance(lm, importance_type='pval')
        """

        feas_imp = pd.DataFrame([(v[0], abs(float(v[3])), float(v[4]))
                                 for v in model.summary().tables[1].data[1:-1]],
                                columns=['feature', 'zscore', 'pval']).set_index('feature')
        feas_imp['occur_time'] = 1

        if isinstance(importance_type, str):
            importance_type = [importance_type]
        if importance_type == ['all']:
            return feas_imp
        else:
            return feas_imp[importance_type]

    @classmethod
    def tree_importance(cls, model, importance_type='all'):
        """
        Return the importance of features in tree models

        :param model: tree model support DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting/Xgboost/Lightgbm/Catboost
        :param importance_type: support str or list like 'all'/'total_gain'/['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        :return:

        Examples:
            (1) feas_imp = tree_importance(model, importance_type='all')
            (2) feas_imp = tree_importance(model, importance_type=['total_gain', 'total_cover'])
        """

        # sklearn tree importance
        if re.search('sklearn.tree|sklearn.ensemble', str(type(model))):
            # If base_estimator in BaggingClassifier is LogisticRegression, features should be normalized
            if re.search('Bagging', str(model)):
                feas_imp = pd.DataFrame({'features': model.feature_names_in_}).reset_index()
                for i in range(model.n_estimators):
                    if re.search('LogisticRegression', str(model.base_estimator_)):
                        tmp = pd.DataFrame({'index': model.estimators_features_[i],
                                            f'imp_{i}': [round(val, 6) for val in
                                                         model.estimators_[i].coef_[0]]}).drop_duplicates()
                    else:
                        tmp = pd.DataFrame({'index': model.estimators_features_[i],
                                            f'imp_{i}': [round(val, 6) for val in
                                                         model.estimators_[i].feature_importances_]}).drop_duplicates()
                        tmp = tmp.groupby('index')[f'imp_{i}'].sum().to_frame(f'imp_{i}').reset_index()
                    feas_imp = feas_imp.merge(tmp, on='index', how='left')

                feas_imp['bagging_importance'] = abs(feas_imp[[c for c in feas_imp if re.search('imp', c)]].sum(axis=1))
                return feas_imp[['features', 'bagging_importance']].set_index('features')

            else:
                feas_imp = pd.DataFrame(model.feature_importances_, index=model.feature_names_in_,
                                        columns=['gini_importance'])
                feas_imp.index.name = 'features'
                return feas_imp

        # booster tree importance
        else:
            if re.search('catboost', str(model)):
                feas_imp = model.get_feature_importance(type='LossFunctionChange',
                                                        data=model.dtrain,
                                                        prettified=True)  # prettified to get dataframe
                feas_imp = feas_imp.fillna(0).rename(
                    columns={'Feature Id': 'features', 'Importances': 'lossFunctionChange_importance'}).set_index(
                    'features')

            elif re.search('xgboost', str(model)):
                for i, type_ in enumerate(['weight', 'gain', 'cover', 'total_gain', 'total_cover']):
                    feas_imp_ = pd.DataFrame.from_dict(model.get_score(importance_type=type_), orient='index',
                                                       columns=[type_])
                    if i == 0:
                        feas_imp = feas_imp_
                    else:
                        feas_imp = feas_imp.merge(feas_imp_, left_index=True, right_index=True)
                    feas_imp.index.name = 'features'

            elif re.search('lightgbm', str(model)):
                for i, type_ in enumerate(['split', 'gain']):
                    feas_imp_ = pd.DataFrame(model.feature_importance(importance_type=type_),
                                             index=model.feature_name(), columns=[type_])
                    if i == 0:
                        feas_imp = feas_imp_
                    else:
                        feas_imp = feas_imp.merge(feas_imp_, left_index=True, right_index=True)
                    feas_imp.index.name = 'features'

            else:
                raise ValueError(
                    'support DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting/Xgboost/Lightgbm/Catboost')

            if isinstance(importance_type, str):
                importance_type = [importance_type]
            if importance_type == ['all']:
                return feas_imp
            else:
                return feas_imp[importance_type]

    @classmethod
    def tree_shap_importance(cls, model, x_train, **kwargs):
        """
        SHAP provides multiple explainers for models.
        TreeExplainer : Support XGBoost, LightGBM, CatBoost and scikit-learn models by Tree SHAP.
        DeepExplainer (DEEP SHAP) : Support TensorFlow and Keras models by using DeepLIFT and Shapley values.
        GradientExplainer : Support TensorFlow and Keras models.
        KernelExplainer (Kernel SHAP) : Applying to any models by using LIME and Shapley values.
        https://shap.readthedocs.io/en/latest/#
        :param model: model
        :param x_train: x_train
        :return:

        Examples:
            (1) tree_shap_importance(model, x_train[feas])
            (2) tree_shap_importance(model, x_train[feas], is_plot=True)
        """

        if re.search('_bagging.BaggingClassifier|_weight_boosting.AdaBoostClassifier', str(type(model))):
            raise ValueError(
                'Model type not yet supported by TreeExplainer: <class sklearn.ensemble._bagging.BaggingClassifier or sklearn.ensemble._weight_boosting.AdaBoostClassifier>')

        # calc shap_values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        if re.search('DecisionTree|RandomForest|ExtraTrees', str(type(model))):
            shap_values = shap_values[0]

        # calc global_shap_values
        max_display, feature_names = len(x_train), list(x_train)
        global_shap_values = np.abs(shap_values).mean(0)
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
        feature_index = feature_order[:max_display]
        shap_imp = pd.DataFrame({'features': [feature_names[i] for i in feature_index], \
                                 'shap_importance': global_shap_values[feature_index]}).sort_values('shap_importance',
                                                                                                    ascending=False). \
            reset_index(drop=True).set_index('features')

        is_plot = kwargs.get('is_plot', False)
        if is_plot:
            shap.summary_plot(shap_values, x_train, plot_type="bar",
                              max_display=shap_imp[shap_imp['shap_importance'] > 0].shape[0])

        return shap_imp

    @classmethod
    def permutation_importance(cls, model, df, target, cv=5):
        """
        Get model permutation importance.Modify feas value to check the performance change

        :param model: model
        :param df: df
        :param target: target
        :param cv: cv
        :return:
        """

        feas = get_feature_names(model)
        # calc base_ks&base_auc
        base_ks, base_auc, _ = evaluate_performance(df[target], ModelApply.model_predict(df[feas], model),
                                                    to_plot=False)

        # shuffle feas
        auc_add_mean, ks_add_mean, feas_list = [], [], []
        for fea in tqdm(feas):
            i, auc_add_list, ks_add_list = 0, [], []
            while i < cv:
                tmp = df[feas + [target]]
                list_ = tmp[fea]
                tmp[fea] = np.random.permutation(list_)
                if str(list_.dtypes) == 'category':
                    tmp[fea] = tmp[fea].astype('category')

                # calc_auc,ks
                calc_ks, calc_auc, _ = evaluate_performance(tmp[target], ModelApply.model_predict(tmp[feas], model),
                                                            to_plot=False)
                auc_add_list.append(base_auc - calc_auc)
                ks_add_list.append(base_ks - calc_ks)
                i = i + 1

            auc_add_mean.append(np.mean(auc_add_list))
            ks_add_mean.append(np.mean(ks_add_list))
            feas_list.append(fea)

        permutation_imp = pd.DataFrame(
            {'permutation_importance_auc': auc_add_mean, 'permutation_importance_ks': ks_add_mean}, index=feas_list)
        permutation_imp['permutation_importance'] = permutation_imp['permutation_importance_auc'] + permutation_imp[
            'permutation_importance_ks']
        permutation_imp.index.names = ['features']
        return permutation_imp  # [['permutation_importance']]


class ModelApply:

    @classmethod
    def lr_fit(cls, x_train, y_train, params=None):
        """
        Fit logistic model with statsmodels.
        x_train, y_train should be pandas type (DataFrame & Series) or numpy.ndarray at the same time.
        https://www.statsmodels.org/dev/_modules/statsmodels/discrete/discrete_model.html

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: fit_regularized params
        :return:

        Examples:
            lm = lr_fit(x_train[feas], x_train[target], params=None)
        """

        if params is None:
            params = {
                'method': 'l1',  # 'l1_cvxopt_cp', 当选择l1的时候，alpha可以选的小一些，当选择l1_cvxopt_cp，alpha可以选的大一些
                'maxiter': 'defined_by_method',  # int, 2000
                # If 'defined_by_method', then use method defaults
                'full_output': 1,
                'callback': None,
                'alpha': 0,  # The weight multiplying the l1 penalty term  l1正则化强度
                'trim_mode': 'auto',
                # If 'auto',trim (set to zero) parameters that would have been zero if the solver reached the theoretical minimum.
                # If 'size', trim params if they have very small absolute value.
                'disp': 0
            }
        X = sm.add_constant(x_train, prepend=False, has_constant='add')
        model = sm.Logit(y_train, X).fit_regularized(**params)

        return model

    @classmethod
    def lr_predict(cls, x_train, model):
        """
        Predict with input model

        :param x_train: DataFrame/Series/numpy.ndarray  x_train
        :param model: lm model
        :return:

        Examples:
            predict = lr_predict(x_train[feas], lm)
        """

        return np.round(model.predict(sm.add_constant(x_train, prepend=False, has_constant='add')), 6)

    @classmethod
    def stepwise_lr(cls, x_train, target, lm, params=None,
                    drop_cnt=0, by='pval', step=1, thresh_pval=0.05):
        """
        Feature selection with stepwise LR, that is, removing features stepwise.

        :param x_train: x_train
        :param target: target
        :param lm: initial LR model
        :param params: fit_regularized params
        :param drop_cnt: number of features to drop when p-value is greater than threshold
        :param by: pval/zscore   drop features with low z-score or high pval, work only when drop_cnt>0
        :param step: number of features to drop each iteration
        :param thresh_pval: p-value threshold
        :returns: LR model, and features kept

        Examples:
            (1) lm, predictions = lr_fit(x_train[feas], x_train[target], model=None)
                stepwise_lr(x_train, target, lm=lm, drop_cnt=0, by='pval', step=1, thresh_pval=0.05)
            (2) lm, predictions = lr_fit(x_train[feas], x_train[target], model=None)
                stepwise_lr(x_train, target, lm=lm, drop_cnt=2, by='pval', step=1, thresh_pval=0.05)
            (3) lm, predictions = lr_fit(x_train[feas], x_train[target], model=None)
                stepwise_lr(x_train, target, lm=lm, drop_cnt=3, by='zscore', step=1, thresh_pval=0.05)
        """

        while True:
            feas_imp = ModelImportance.lr_importance(lm, importance_type='all')
            if by == 'permutation_importance':
                permutation_imp = ModelImportance.permutation_importance(lm, x_train, target)
                feas_imp = feas_imp.merge(permutation_imp, left_index=True, right_index=True)

            idx_pval_na = feas_imp['pval'].isnull()
            if idx_pval_na.sum() > 0:
                var_lr = feas_imp.loc[~idx_pval_na].index.tolist()

            elif np.sum(feas_imp['pval'] > thresh_pval):
                feas_imp = feas_imp.sort_values('pval', ascending=False)
                step_cur = min(step, np.sum(feas_imp['pval'] > thresh_pval))
                var_lr = feas_imp.iloc[step_cur:].index.tolist()

            elif drop_cnt > 0:
                if by == 'pval':
                    feas_imp = feas_imp.sort_values('pval', ascending=False)
                elif by in ['zscore', 'permutation_importance']:
                    feas_imp = feas_imp.sort_values(by, ascending=True)
                else:
                    raise ValueError('Unknown by')
                step_cur = min(step, drop_cnt)
                var_lr = feas_imp.iloc[step_cur:].index.tolist()
                drop_cnt -= step_cur

            else:
                return lm, get_feature_names(lm)

            lm = cls.lr_fit(x_train[var_lr].astype(float), x_train[target], params=params)

    @classmethod
    def DecisionTree_fit(cls, x_train, y_train, params=None):
        """
        Fit decision tree model

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: decisiontree params
        :return:

        Examples:
            model = DecisionTree_fit(x_train[feas], x_train[target], params=None)
        """

        if params is None:
            params = {
                'criterion': 'entropy',  # The function to measure the quality of a split. Supported criteria are
                # "gini" for the Gini impurity and "entropy" for the information gain.
                # 一般entropy生成的叶节点会更多一些，gini会少一些
                'splitter': 'best',  # The strategy used to choose the split at each node. Supported strategies are
                # "best" to choose the best split and "random" to choose the best random split.
                'max_depth': 5,
                'min_samples_split': 0.05,  # 1000 The minimum number of samples required to split an internal node
                'min_samples_leaf': 0.02,  # 200 #  The minimum number of samples required to be at a leaf node.
                #  If int, then consider `min_samples_leaf` as the minimum number.
                #  If float, then `min_samples_leaf` is a fraction and ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

                'min_weight_fraction_leaf': 0,  # The minimum weighted fraction of the sum total of weights(权重总和的最小加权分数)
                'max_features': None,  # int, float or {"auto", "sqrt", "log2"}, default=None
                'max_leaf_nodes': None,
                # Grow a tree with ``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity.
                # If None then unlimited number of leaf nodes.
                'min_impurity_decrease': 0.00002,
                # A node will be split if this split induces a decrease of the impurity
                # min_impurity_decrease 一般都设置的很小，比如=0或者=0.00002
                'class_weight': None,  # balanced,
                'ccp_alpha': 0,  # 代价复杂度减枝

                'random_state': 2020
            }

        model = tree.DecisionTreeClassifier(**params)
        _ = model.fit(x_train, y_train)

        return model

    @classmethod
    def RandomForest_fit(cls, x_train, y_train, params=None):
        """
        Fit RandomForest model

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: RandomForest params
        :return:

        Examples:
            model = RandomForest_fit(x_train[feas], x_train[target]), params=None)
        """

        if params is None:
            params = {
                'n_estimators': 2000,  # The number of trees in the forest.
                'criterion': 'entropy',  # entropy "gini" for the Gini impurity and "entropy" for the information gain.
                'max_depth': 4,  # The maximum depth of the tree. If None, then nodes are expanded until
                'min_samples_split': 0.05,  # If int, then consider `min_samples_split` as the minimum number.
                # If float, then `min_samples_split` is a fraction and`ceil(min_samples_split * n_samples)` are the minimum number of samples for each split.
                'min_samples_leaf': 0.02,  # 2000
                'min_weight_fraction_leaf': 0.01,  # min_weight_fraction_leaf must in [0, 0.5]
                # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node
                # 叶节点所需的权重总和（所有输入样本的）的最小加权分数(注：设置的小一些要不然很容易欠拟合)
                'max_features': 'auto',  # {"auto", "sqrt", "log2"}, int or float, default="auto"
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0.00002,
                # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

                'warm_start': False,
                # When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
                'ccp_alpha': 0,  # 代价复杂度剪枝
                'max_samples': 0.8,  # 60000， 样本采样

                'random_state': 2020
            }

        model = RandomForestClassifier(**params)
        _ = model.fit(x_train, y_train)

        return model

    @classmethod
    def ExtraTrees_fit(cls, x_train, y_train, params=None):
        """
        Fit ExtraTrees model

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: ExtraTrees params
        :return:

        Examples:
            model = ExtraTrees_fit(x_train[feas], x_train[target]), params=None)
        """

        if params is None:
            params = {
                'n_estimators': 1500,
                'criterion': 'entropy',  # {"gini", "entropy"}, default="gini"
                'max_depth': 8,
                'min_samples_split': 0.02,
                'min_samples_leaf': 0.01,
                'min_weight_fraction_leaf': 0,  # The minimum weighted fraction of the sum total of weights
                'max_samples': 0.9,  # 样本采样情况
                'ccp_alpha': 0.0,

                'max_features': 'auto',
                'min_impurity_decrease': 0,
                'bootstrap': True,  # 在随机选取样本时是否进行重置
                'warm_start': True,
                'random_state': 2020
            }

        model = ExtraTreesClassifier(**params)
        _ = model.fit(x_train, y_train)

        return model

    @classmethod
    def Bagging_fit(cls, x_train, y_train, params=None):
        """
        Fit Bagging model warning: trainning time is very long!
        base_estimator can have big n_estimators(be a strong classifier) but BaggingClassifier do not need big n_estimators
        is not a very good model but very stable. DecisionTreeClassifier or LogisticRegression as base_estimator always having relatively good performance

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: Bagging params
        :return:

        Examples:
            (1) model = Bagging_fit(x_train[feas], x_train[target]), params=None)  # use DecisionTreeClassifier as base_estimator

            (2) params = {
                    'base_estimator': RandomForestClassifier(**{
                                                                'n_estimators': 1500,
                                                                'criterion': 'entropy',
                                                                'max_depth': 6,
                                                                'min_samples_split': 0.06,
                                                                'min_samples_leaf': 0.03,
                                                                'min_weight_fraction_leaf': 0,
                                                                'max_features': 'auto',
                                                                'max_leaf_nodes': None,
                                                                'min_impurity_decrease': 0.00002,
                                                                'warm_start': True,
                                                                'max_samples': 0.9,
                                                                'random_state': 2020
                                                                }),
                    'n_estimators': 10,  #  The number of base estimators in the ensemble
                    'max_samples': 0.8,   # 样本采样情况
                    'max_features': 0.9,  # 特征采样情况
                                          # If int, then draw `max_samples` samples.  If float, then draw `max_samples * X.shape[0]` samples.
                    'bootstrap':True,  # Whether samples are drawn with replacement. 在随机选取样本时是否进行重置
                    'bootstrap_features':True, # Whether features are drawn with replacement. 在随机选取特征时是否进行重置
                    'warm_start': True,
                    'random_state':2020
                        }
                model = Bagging_fit(x_train[feas], x_train[target]), params=params)  # use RandomForestClassifier as base_estimator

            (3) params = {
                    'base_estimator': LogisticRegression(**{
                                                            'penalty': 'l2',  # {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
                                                                                       # 'l2'`: add a L2 penalty term and it is the default choice;
                                                                                       # `'l1'`: add a L1 penalty term;
                                                                                       # `'elasticnet'`: both L1 and L2 penalty terms are added.
                                                            'tol': 0.0001, # 停止标准的容差
                                                            'C': 3, # 惩罚项，
                                                            'solver': 'lbfgs',  # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},     default='lbfgs'
                                                                                    # For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones;
                                                                                    # Solver newton-cg/lbfgs supports only 'l2' or 'none' penalties, got elasticnet penalty.
                                                            'max_iter': 100,  # Maximum number of iterations taken for the solvers to converge.
                                                            'warm_start': True,
                                                            'random_state':2020,
                                                             }),

                    'n_estimators': 25,  #  The number of base estimators in the ensemble
                    'max_samples': 0.8,   # 样本采样情况
                    'max_features': 0.9,  # 特征采样情况
                                          # If int, then draw `max_samples` samples.  If float, then draw `max_samples * X.shape[0]` samples.
                    'bootstrap':True,  # Whether samples are drawn with replacement. 在随机选取样本时是否进行重置
                    'bootstrap_features':True, # Whether features are drawn with replacement. 在随机选取特征时是否进行重置
                    'warm_start': True,

                    'random_state':2020
                            }
                model = Bagging_fit(x_train[feas], x_train[target]), params=params)  # use LogisticRegression as base_estimator

            (4) params = {
                    'base_estimator': ExtraTreesClassifier(**{
                                                            'n_estimators':1500,
                                                            'criterion':'entropy',
                                                            'max_depth':7,
                                                            'min_samples_split':0.02,
                                                            'min_samples_leaf':0.01,
                                                            'min_weight_fraction_leaf':0,
                                                            'max_samples': 0.9,
                                                            'ccp_alpha': 0.0,
                                                            'max_features':'auto',
                                                            'min_impurity_decrease':  0,
                                                            'bootstrap':True,
                                                            'warm_start': True,
                                                            'random_state':2020
                                                            }),

                    'n_estimators': 10,  #  The number of base estimators in the ensemble
                    'max_samples': 0.8,   # 样本采样情况
                    'max_features': 0.9,  # 特征采样情况
                                          # If int, then draw `max_samples` samples.  If float, then draw `max_samples * X.shape[0]` samples.
                    'bootstrap':True,  # Whether samples are drawn with replacement. 在随机选取样本时是否进行重置
                    'bootstrap_features':True, # Whether features are drawn with replacement. 在随机选取特征时是否进行重置
                    'warm_start': True,

                    'random_state':2020
                        }
                model = Bagging_fit(x_train[feas], x_train[target]), params=params)  # use ExtraTreesClassifier as base_estimator

        """

        if params is None:
            params = {
                'base_estimator': tree.DecisionTreeClassifier(**{
                    'criterion': 'entropy',
                    'splitter': 'best',
                    'max_depth': 6,
                    'min_samples_split': 0.02,
                    'min_samples_leaf': 0.01,
                    'random_state': 2020}
                                                              ),
                'n_estimators': 400,  # The number of base estimators in the ensemble
                'max_samples': 0.8,  # 样本采样情况
                'max_features': 0.9,
                # 特征采样情况  # If int, then draw `max_samples` samples.  If float, then draw `max_samples * X.shape[0]` samples.
                'bootstrap': True,  # Whether samples are drawn with replacement. 在随机选取样本时是否进行重置
                'bootstrap_features': True,  # Whether features are drawn with replacement. 在随机选取特征时是否进行重置
                'warm_start': True,

                'random_state': 2020
            }

        model = BaggingClassifier(**params)
        _ = model.fit(x_train, y_train)

        return model

    @classmethod
    def Adaboost_fit(cls, x_train, y_train, params=None):
        """
        Fit Adaboost model
        base_estimator can be a weak classifier

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: Adaboost params
        :return:

        Examples:
            (1) model = Adaboost_fit(x_train[feas], x_train[target]), params=None)  # use DecisionTreeClassifier as base_estimator
            (2) params = {
                        'base_estimator': RandomForestClassifier(**{
                                                                    'n_estimators': 100,
                                                                    'criterion': 'entropy',
                                                                    'max_depth': 3,
                                                                    'min_samples_split': 0.06,
                                                                    'min_samples_leaf': 0.03,
                                                                    'min_weight_fraction_leaf': 0,
                                                                    'max_features': 'auto',
                                                                    'max_leaf_nodes': None,
                                                                    'min_impurity_decrease': 0.00002,
                                                                    'warm_start': True,
                                                                    'max_samples': 0.9,
                                                                    'random_state': 2020
                                                                    }),

                        'n_estimators': 30,  #  The number of base estimators in the ensemble
                        'learning_rate':  0.5,  #  Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier
                                                # 在每次提升迭代中应用于每个分类器的权重。 一个更高的学习率增加了每个分类器的贡献
                        'algorithm':'SAMME.R',  #  {'SAMME', 'SAMME.R'}, default='SAMME.R'
                                                # The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

                        'random_state':2020
                        }
                model = Adaboost_fit(x_train[feas], x_train[target]), params=None)  # use RandomForestClassifier as base_estimator

            (3) params = {
                        'base_estimator': ExtraTreesClassifier(**{
                                                                'n_estimators':100,
                                                                'criterion':'entropy', # {"gini", "entropy"}, default="gini"
                                                                'max_depth':3,
                                                                'min_samples_split':0.05,
                                                                'min_samples_leaf':0.03,
                                                                'min_weight_fraction_leaf':0,  # The minimum weighted fraction of the sum total of weights
                                                                'max_samples': 0.9,   # 样本采样情况
                                                                'ccp_alpha': 0.0,

                                                                'max_features':'auto',
                                                                'min_impurity_decrease':  0,
                                                                'bootstrap':True,  # 在随机选取样本时是否进行重置
                                                                'warm_start': True,
                                                                'random_state':2020
                                                                }),

                        'n_estimators': 30,  #  The number of base estimators in the ensemble
                        'learning_rate':  0.5,  #  Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier
                                                # 在每次提升迭代中应用于每个分类器的权重。 一个更高的学习率增加了每个分类器的贡献
                        'algorithm':'SAMME.R',  #  {'SAMME', 'SAMME.R'}, default='SAMME.R'
                                                # The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

                        'random_state':2020
                        }
                model = Adaboost_fit(x_train[feas], x_train[target]), params=None)  # use ExtraTreesClassifier as base_estimator
        """

        if params is None:
            params = {
                'base_estimator': tree.DecisionTreeClassifier(**{
                    'criterion': 'entropy',
                    'splitter': 'best',
                    'max_depth': 3,
                    'min_samples_split': 0.08,
                    'min_samples_leaf': 0.05,
                    'random_state': 2020
                }),
                'n_estimators': 30,  # The number of base estimators in the ensemble
                'learning_rate': 0.5,
                # Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier
                # 在每次提升迭代中应用于每个分类器的权重。 一个更高的学习率增加了每个分类器的贡献
                'algorithm': 'SAMME.R',  # {'SAMME', 'SAMME.R'}, default='SAMME.R'
                # The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.

                'random_state': 2020
            }

        model = AdaBoostClassifier(**params)
        _ = model.fit(x_train, y_train)

        return model

    @classmethod
    def GradientBoosting_fit(cls, x_train, y_train, params=None):
        """
        Fit gradientBoosting model

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param params: gradientBoosting params
        :return:

        Examples:
            model = GradientBoosting_fit(x_train[feas], x_train[target]), params=None)
        """

        if params is None:
            params = {
                'loss': 'deviance',  # {'deviance', 'exponential'}, default='deviance'
                # 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs.
                # For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.
                'max_depth': 3,
                'learning_rate': 0.01,
                'n_estimators': 1000,

                'subsample': 0.8,
                'criterion': 'friedman_mse',
                # {'friedman_mse', 'squared_error', 'mse', 'mae'},    default='friedman_mse'
                #  The default value of 'friedman_mse' is generally the best as it can provide a better approximation in some cases.
                # 'squared_error' for mean squared error, and 'mae' for the mean absolute error
                'min_samples_split': 0.05,
                'min_samples_leaf': 0.02,
                'min_weight_fraction_leaf': 0,
                'min_impurity_decrease': 0.00001,

                'max_features': None,  # {'auto', 'sqrt', 'log2'}, int or float, default=None
                'warm_start': True,
                'n_iter_no_change': 50,  # early_stop
                'ccp_alpha': 0,

                'random_state': 2020}

        model = GradientBoostingClassifier(**params)
        _ = model.fit(x_train, y_train)

        # add model information
        model.params = params

        return model

    @classmethod
    def XGBoost_fit(cls, x_train, y_train, x_test, y_test, params=None):
        """
        Fit XGBoost model
        # https://xgboost.readthedocs.io/en/latest/parameter.html#:~:text=XGBoost%20Parameters.%20%C2%B6.%20Before%20running%20XGBoost%2C%20we%20must,parameters%20depend%20on%20which%20booster%20you%20have%20chosen.

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param x_test:  DataFrame/Series/numpy.ndarray  x_test
        :param y_test:  DataFrame/Series/numpy.ndarray  y_test
        :param params: xgboost params
        :return:

        Examples:
            model = XGBoost_fit(x_train[feas], x_train[target], x_test[feas], x_test[target], params=None)
        """

        if params is None:
            params = {
                'booster': 'gbtree',  # gbtree/gblinear/dart
                'objective': 'reg:logistic',
                'eval_metric': ['auc'],

                'learning_rate': 0.03,
                'gamma': 2,
                # Minimum loss reduction required to make a further partition on a leaf node of the tree
                'max_depth': 3,
                'min_child_weight': 30,  # Minimum sum of instance weight (hessian) needed in a child
                'subsample': 0.8,
                'sampling_method': 'uniform',  # gradient_based
                'colsample_bytree': 0.9,  # the subsample ratio of columns when constructing each tree
                'colsample_bylevel': 0.8,  # is the subsample ratio of columns for each level
                'colsample_bynode': 0.8,  # is the subsample ratio of columns for each node
                'lambda': 1,  # L2 regularization term on weights
                'alpha': 1,  # L1 regularization term on weights

                # 'scale_pos_weight': 29.5, # Control the balance of positive and negative weights, useful for unbalanced classes
                # (x_train[target]==0).sum()/(x_train[target]==1).sum()
                'grow_policy': 'lossguide',  # lossguide: split at nodes with highest loss change.（在损失变化最大的节点处分裂）
                # depthwise: split at nodes closest to the root.（在最靠近根的节点处分裂）
                'iterations': 600,
                'early_stopping_rounds': 50,
                'nthread': 1,
                'seed': 2021,
                'verbose_eval': 100

                #          # only use when booster is dart
                #          'sample_type': 'uniform',   # uniform: dropped trees are selected uniformly.
                #                                       # weighted: dropped trees are selected in proportion to weight.
                #          'normalize_type': 'tree',    # tree: new trees have the same weight of each of dropped trees.
                #                                       # forest: new trees have the same weight of sum of dropped trees (forest).
                #          'rate_drop': 0.15,           # Dropout rate
                #          'skip_drop': 0.1             # Probability of skipping the dropout procedure during a boosting iteration.

            }

        result = {}
        # get dtrain/dtest
        dtrain = xgb.DMatrix(x_train, y_train)
        if x_test is None:
            dtest = dtrain
        else:
            dtest = xgb.DMatrix(x_test, y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_boost_round=params.get('iterations', 2000),
                          early_stopping_rounds=params.get('early_stopping_rounds', 50),
                          evals=watchlist, verbose_eval=params.get('verbose_eval', 100), evals_result=result)

        # add model information
        model.params = params
        model.result = result

        return model

    @classmethod
    def LightGBM_fit(cls, x_train, y_train, x_test, y_test, params=None):
        """
        Fit LightGBM model
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param x_test:  DataFrame/Series/numpy.ndarray  x_test
        :param y_test:  DataFrame/Series/numpy.ndarray  y_test
        :param params: lightGBM params
        :return:

        Examples:
            model = LightGBM_fit(x_train[feas], x_train[target], x_test[feas], x_test[target], params=None)
        """

        if params is None:
            params = {'objective': 'regression',
                      'metric': ['auc'],
                      'boosting': 'gbdt',
                      # gbdt/rf/dart/goss(Gradient-based One-Side Sampling) 使用dart一般iterations和learning_rate可以设的大一点,不太会过拟合
                      'iterations': 350,
                      'learning_rate': 0.02,
                      'max_depth': -1,
                      'num_leaves': 2 ** 7 - 1,
                      'min_data_in_leaf': 200,
                      'min_child_weight': 0.01,  # minimal sum hessian in one leaf (lgb 是sum in leaf)
                      'tree_learner': 'serial',
                      # serial/feature(feature_parallel特征并行)/data(data_parallel/数据并行)/voting(voting_parallel选举并行)

                      'force_col_wise': False,  # 强制按列建立直方图
                      'force_row_wise': False,  # 强制按行建立直方图

                      'bagging_fraction': 0.8,  # 即subsample
                      'bagging_freq': 5,
                      # k means perform bagging at every k iteration(can not use when boosting is goss)
                      'feature_fraction': 0.7,  # 即colsample_bytree
                      'feature_fraction_bynode': 0.7,  # 即colsample_bynode
                      'extra_trees': False,  # use extremely randomized trees

                      'early_stopping_rounds': 30,
                      'lambda_l1': 25,
                      'lambda_l2': 40,
                      'min_gain_to_split': 0.01,  # the minimal gain to perform split

                      'min_data_per_group': 70,  # minimal number of data per categorical group
                      'max_cat_threshold': 10,  # limit number of split points considered for categorical features
                      'cat_l2': 10,  # L2 regularization in categorical split
                      'cat_smooth': 10,
                      # this can reduce the effect of noises in categorical features, especially for categories with few data
                      'max_cat_to_onehot': 4,  # 如果cat特征的取值<=max_cat_to_onehot,则切分方式为one-vs-rest，否则为many-vs-many

                      'max_bin': 20,  # max number of bins that feature values will be bucketed in(即直方图的bin)
                      'min_data_in_bin': 50,  # minimal number of data inside one bin

                      'enable_bundle': True,  # set this to false to disable Exclusive Feature Bundling (EFB互斥特征捆绑算法)
                      'seed': 2021,
                      'verbose': -1,
                      'verbose_eval': 100,

                      # only use when booster is dart
                      # 'drop_rate': 0.1,
                      # 'max_drop': 2,
                      # 'skip_drop': 0.1,
                      # 'uniform_drop': False,  # set this to true, if you want to use uniform drop

                      # # only use when booster is goss
                      # 'top_rate': 0.2, # the retain ratio of large gradient data (单边梯度采样中，信息增益大的样本)
                      # 'other_rate': 0.1, # the retain ratio of small gradient data (单边梯度采样中，信息增益小的样本)

                      # # only use when tree_learner is voting
                      # 'top_k': 25, # 即选举并行中的K
                      }

        result = {}
        # get dtrain/dtest
        dtrain = lgb.Dataset(x_train, y_train)
        if x_test is None:
            dtest = dtrain
        else:
            dtest = lgb.Dataset(x_test, y_test, reference=dtrain)

        # fit
        model = lgb.train(params, dtrain, verbose_eval=params.get('verbose_eval', 100),
                          num_boost_round=params.get('iterations', 500),
                          valid_sets=[dtrain, dtest], valid_names=['train', 'test'], evals_result=result)

        # add model information
        model.params = params
        model.result = result

        return model

    @classmethod
    def Catboost_fit(cls, x_train, y_train, x_test, y_test, params=None):
        """
        Fit Catboost model
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html

        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param x_test:  DataFrame/Series/numpy.ndarray  x_test
        :param y_test:  DataFrame/Series/numpy.ndarray  y_test
        :param params: Catboost params
        :return:

        Examples:
            model = Catboost_fit(x_train[feas], x_train[target], x_test[feas], x_test[target], params=None)
        """

        if params is None:
            params = {
                'loss_function': 'Logloss',
                'custom_loss': 'AUC',
                'eval_metric': 'AUC',
                'iterations': 300,
                'learning_rate': 0.03,
                'l2_leaf_reg': 15,
                'random_strength': 5,
                # The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.

                'use_best_model': True,
                'max_depth': 6,
                #          'min_data_in_leaf': 200, # Can be used only with the Lossguide and Depthwisegrowing policies.
                #          'max_leaves':2 ** 4 - 1, # Can be used only with the Lossguide growing policy.
                'leaf_estimation_method': 'Gradient',
                # The method used to calculate the values in leaves. Newton/Gradient
                'leaf_estimation_backtracking': 'AnyImprovement',
                # The type of backtracking to use during the gradient descent. AnyImprovement/No

                'grow_policy': 'SymmetricTree',  # catboost的一大特点之一，catboost默认是对称树 SymmetricTree
                # SymmetricTree/Depthwise(类似于xgb)/Lossguide(类似于lgb)
                # Depthwise 和 Lossguide 不支持Feature importance and ShapValues

                'boosting_type': 'Ordered',  # cagboost的另一大特点 梯度偏差优化 ordered boosting
                # Ordered： Usually provides better quality on small datasets, but it may be slower than the Plain scheme
                # Plain： The classic gradient boosting scheme

                'nan_mode': 'Min',  # Forbidden/Min/Max
                'one_hot_max_size': 4,  # catboost将会对所有unique值<=one_hot_max_size的特征进行独热处理

                'bootstrap_type': 'Bernoulli',  # Bayesian、Bernoulli、MVS、Poisson (supported for GPU only)、No
                #          'bagging_temperature':5, # Defines the settings of the Bayesian bootstrap, The higher the value the more aggressive the bagging is
                'subsample': 0.9,  # Sample rate for bagging only use when bootstrap types is Poisson or Bernoulli
                'sampling_frequency': 'PerTreeLevel',
                # Frequency to sample weights and objects when building trees. PerTree/PerTreeLevel

                'random_seed': 2021,
                'verbose_eval': 100,
                'od_wait': 50
            }

        # get dtrain/dtest
        cat_feas = x_train.select_dtypes('object').columns.tolist()
        dtrain = Pool(data=x_train, label=y_train, cat_features=cat_feas)
        if x_test is None:
            dtest = dtrain
        else:
            dtest = Pool(data=x_test, label=y_test, cat_features=cat_feas)

        # fit
        model = ctb.train(dtrain=dtrain, params=params, iterations=params.get('iterations', 500),
                          eval_set=[dtrain, dtest], verbose=params.get('verbose_eval', 100),
                          early_stopping_rounds=params.get('od_wait', 50), plot=False)

        # add model information
        model.params = params
        model.dtrain = dtrain
        model.best_iteration = model.best_iteration_
        results = model.get_evals_result()
        results['train'] = results.pop('validation_0')
        results['test'] = results.pop('validation_1')
        model.result = results

        return model

    @classmethod
    def tree_fit(cls, model_type, x_train, y_train, x_test=None, y_test=None, params=None):
        """
        Fit tree model. support DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/BaggingClassifier/AdaBoostClassifier/GradientBoostingClassifier/xgboost/lightgbm/catboost

        :param model_type:  model_type support DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/BaggingClassifier/AdaBoostClassifier/GradientBoostingClassifier/xgboost/lightgbm/catboost
        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param x_test:  DataFrame/Series/numpy.ndarray  x_test
        :param y_test:  DataFrame/Series/numpy.ndarray  y_test
        :param params: tree model params
        :return:

        Examples:
            if re.search('sklearn.tree|sklearn.ensemble', str(type(model))):
                model_type = str(model).split('(')[0]
            else:
                model_type = str(model).split('.')[0][1:]

            (1) model = tree_fit('DecisionTreeClassifier', x_train[feas], x_train[target], params=None)
            (2) model = tree_fit('RandomForestClassifier', x_train[feas], x_train[target], params=None)
            (3) model = tree_fit('xgboost',x_train[feas], x_train[target], x_test[feas], x_test[target], params=None)
        """

        if model_type == 'DecisionTreeClassifier':
            print('start fit DecisionTree model')
            return cls.DecisionTree_fit(x_train, y_train, params)

        elif model_type == 'RandomForestClassifier':
            print('start fit RandomForest model')
            return cls.RandomForest_fit(x_train, y_train, params)

        elif model_type == 'ExtraTreesClassifier':
            print('start fit ExtraTrees model')
            return cls.ExtraTrees_fit(x_train, y_train, params)

        elif model_type == 'BaggingClassifier':
            print('start fit Bagging model')
            return cls.Bagging_fit(x_train, y_train, params)

        elif model_type == 'AdaBoostClassifier':
            print('start fit AdaBoost model')
            return cls.Adaboost_fit(x_train, y_train, params)

        elif model_type == 'GradientBoostingClassifier':
            print('start fit GradientBoosting model')
            return cls.GradientBoosting_fit(x_train, y_train, params)

        elif model_type == 'xgboost':
            print('start fit Xgboost model')
            return cls.XGBoost_fit(x_train, y_train, x_test, y_test, params)

        elif model_type == 'lightgbm':
            print('start fit Lightgbm model')
            return cls.LightGBM_fit(x_train, y_train, x_test, y_test, params)

        elif model_type == 'catboost':
            print('start fit Catboost model')
            return cls.Catboost_fit(x_train, y_train, x_test, y_test, params)

        else:
            raise ValueError(
                'model_type only support DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/BaggingClassifier/AdaBoostClassifier/GradientBoostingClassifier/xgboost/lightgbm/catboost')

    @classmethod
    def tree_predict(cls, x_train, model):
        """
        Predict with input tree model

        :param x_train: DataFrame/Series/numpy.ndarray  x_train
        :param model: tree model
        :return:

        Examples:
            predict = tree_predict(x_train[feas], model)
        """

        # predict sklearn tree model
        if re.search('sklearn.tree|sklearn.ensemble', str(type(model))):
            return pd.Series(np.round(val[1], 6) for val in model.predict_proba(x_train))

        # predict xgboost model
        elif re.search('xgboost', str(model)):
            return pd.Series(model.predict(xgb.DMatrix(x_train)))

        # predict lightgbm model
        elif re.search('lightgbm', str(model)):
            return pd.Series(model.predict(x_train))

        # predict catboost model
        elif re.search('catboost', str(model)):
            return pd.Series(c[1] for c in model.predict(x_train, prediction_type='Probability'))

        else:
            raise ValueError(
                'only support DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting/Xgboost/Lightgbm/Catboost model')

    @classmethod
    def stepwise_tree(cls, x_train, x_test, target, model, params=None,
                      drop_cnt=0, by='gini_importance', step=10, thresh_imp=0):
        """
        Feature selection with stepwise tree, removing features stepwise.

        :param x_train: x_train
        :param x_test: x_test
        :param target: target
        :param model: initial tree model
        :param params: tree model params
        :param drop_cnt: number of features to drop when importance less than threshold
        :param by:  drop features with low importance, work only when drop_cnt>0
        :param step: number of features to drop each iteration
        :param thresh_imp: importance-value threshold
        :returns: model, and features kept

        Examples:
            (1) model, feas = stepwise_tree(x_train, None, target, model, params=None, drop_cnt=0, by='gini_importance', step=1, thresh_imp=0)
            (2) model, feas = stepwise_tree(x_train, x_test, target, model, params=None, drop_cnt=2, by='total_gain', step=1, thresh_imp=0.01)
            (3) model, feas = stepwise_tree(x_train, x_test, target, model, params=None, drop_cnt=2, by='shap_imp', step=1, thresh_imp=0.01)
        """

        while True:
            if by == 'shap_importance':
                feas_imp = ModelImportance.tree_shap_importance(model, x_train[get_feature_names(model)])
            elif by == 'permutation_importance':
                feas_imp = ModelImportance.permutation_importance(model, x_train, target)
            else:
                feas_imp = ModelImportance.tree_importance(model, importance_type=by)

            idx_imp_na = feas_imp[by].isnull()
            if idx_imp_na.sum() > 0:
                var = feas_imp.loc[~idx_imp_na].index.tolist()

            elif np.sum(feas_imp[by] <= thresh_imp):
                feas_imp = feas_imp.sort_values(by)
                step_cur = min(step, np.sum(feas_imp[by] <= thresh_imp))
                var = feas_imp.iloc[step_cur:].index.tolist()

            elif drop_cnt > 0:
                feas_imp = feas_imp.sort_values(by)
                step_cur = min(step, drop_cnt)
                var = feas_imp.iloc[step_cur:].index.tolist()
                drop_cnt -= step_cur

            else:
                return model, get_feature_names(model)

            # loop fitting
            if re.search('sklearn.tree|sklearn.ensemble', str(type(model))):
                model_type = str(model).split('(')[0]
            else:
                model_type = str(model).split('.')[0][1:]

            if x_test is None:
                model = cls.tree_fit(model_type, x_train[var], x_train[target], x_test=None, y_test=None, params=params)
            else:
                model = cls.tree_fit(model_type, x_train[var], x_train[target], x_test=x_test[var],
                                     y_test=x_test[target], params=params)

    @classmethod
    def tree_apply(cls, df, model, is_dummy=False):
        """
        Indexes of leafs to which dataframe are mapped by model trees.

        :param df: dataframe
        :param model: model
        :param is_dummy: whether to dummy
        :return:

        Examples:
            (1) tmp = tree_apply(x_train[['target']+feas], model, is_dummy=True)
            (2) tmp = tree_apply(x_train, model, is_dummy=False)
        """

        if re.search('DecisionTreeClassifier', str(model)):
            df_apply = pd.DataFrame(model.apply(df[get_feature_names(model)])).rename(columns={0: 'tree_node'}).astype(
                int)

        elif re.search('RandomForestClassifier|ExtraTreesClassifier', str(model)):
            df_apply = pd.DataFrame(model.apply(df[get_feature_names(model)]),
                                    columns=[f'tree_{num}_node' for num in range(model.n_estimators)]).astype(int)

        elif re.search('GradientBoostingClassifier', str(model)):
            df_apply = pd.DataFrame(model.apply(df[get_feature_names(model)]).reshape(-1, model.n_estimators),
                                    columns=[f'tree_{num}_node' for num in range(model.n_estimators)]).astype(int)

        elif re.search('xgboost', str(model)):
            df_apply = pd.DataFrame(model.predict(xgb.DMatrix(df[get_feature_names(model)]), pred_leaf=True),
                                    columns=[f'tree_{num}_node' for num in
                                             range(model.params['iterations'])]).astype(int)

        elif re.search('lightgbm', str(model)):
            cat_feas = df[get_feature_names(model)].select_dtypes('object').columns.tolist()
            df[cat_feas] = df[cat_feas].astype('category')
            df_apply = pd.DataFrame(model.predict(df[get_feature_names(model)], pred_leaf=True),
                                    columns=[f'tree_{num}_node' for num in
                                             range(model.params['iterations'])]).astype(int)

        else:
            raise ValueError(
                'only support DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/GradientBoostingClassifier/xgboost/lightgbm')

        if is_dummy:
            df_apply = pd.get_dummies(df_apply.astype(str))

        df = df.merge(df_apply, left_index=True, right_index=True)
        return df

    @classmethod
    def model_fit(cls, model_type, x_train, y_train, x_test=None, y_test=None, params=None):
        """
        Fit lr/tree model. support LogisticRegression/DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/BaggingClassifier/AdaBoostClassifier/GradientBoostingClassifier/xgboost/lightgbm/catboost

        :param model_type:  model_type support LogisticRegression/DecisionTreeClassifier/RandomForestClassifier/ExtraTreesClassifier/BaggingClassifier/AdaBoostClassifier/GradientBoostingClassifier/xgboost/lightgbm/catboost
        :param x_train:  DataFrame/Series/numpy.ndarray  x_train
        :param y_train:  DataFrame/Series/numpy.ndarray  y_train
        :param x_test:  DataFrame/Series/numpy.ndarray  x_test
        :param y_test:  DataFrame/Series/numpy.ndarray  y_test
        :param params: model params
        :return:

        Examples:
            (1) model = model_fit('LogisticRegression', x_train[feas], x_train[target], params=None)
            (2) model = model_fit('lightgbm', x_train[feas], x_train[target], x_test[feas], x_test[target], params=None)
        """

        if model_type == 'LogisticRegression':
            print('start fit LogisticRegression model')
            return cls.lr_fit(x_train, y_train, params)

        else:
            return cls.tree_fit(model_type, x_train, y_train, x_test, y_test, params)

    @classmethod
    def model_predict(cls, x_train, model):
        """
        Predict with input lr/tree model

        :param x_train: DataFrame/Series/numpy.ndarray  x_train
        :param model:  model
        :return:

        Examples:
            predict = model_predict(x_train[feas], model)
        """

        if re.search('statsmodels', str(model)):
            return cls.lr_predict(x_train, model)
        else:
            return cls.tree_predict(x_train, model)

    @classmethod
    def plot_model_metric(cls, model, metric):
        """
        Plot model metric per iteration

        :param model: model support xgboost/lightgbm/catboost
        :param metric: metric in model support auc/AUC/etc.
        :return:

        Examples:
            (1) # xgboost/lightgbm model
                plot_model_metric(model, metric='auc')
            (2) # catboost model
                plot_model_metric(model, metric='AUC')
        """

        results = model.result
        best_iteration = list(range(model.params['iterations']))[model.best_iteration - 1]

        fig = plt.figure(figsize=(20, 10))
        epochs = len(results['train'][metric])
        x_axis = range(0, epochs)
        plt.plot([best_iteration, best_iteration],  # plot best_iteration
                 [max(results['train'][metric][best_iteration], results['test'][metric][best_iteration]) + 0.01,
                  min(results['train'][metric][0], results['test'][metric][0])],
                 color='k', linewidth=2, label='best_iteration')
        plt.plot(x_axis, results['train'][metric], label='train', linewidth=3)  # plot train metric
        plt.plot(x_axis, results['test'][metric], label='test', linewidth=3)    # plot test metric

        plt.legend(fontsize=20)
        plt.ylabel(f'{metric}', fontsize=25, loc='top')
        plt.tick_params(labelsize=20)
        plt.xlabel('iteration', fontsize=20, loc='right')
        plt.title(f'model_{metric}', fontsize=25)

        plt.close(fig)
        return fig


class DumpTree:

    @classmethod
    def dump_sklearn_tree(cls, model, tree_num, feature_names):
        """
        Show rules from sklearn tree model

        :param model: model
        :param tree_num: tree_num in RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting
        :param feature_names:  feature_names
        :return:

        Examples:
            (1) dump_sklearn_tree(model, tree_num=0, feature_names=mdl.get_feature_names(model))
            (2) dump_sklearn_tree(model, tree_num=90, feature_names=mdl.get_feature_names(model))
            (3) # bagging
                tree_num = 10
                dump_sklearn_tree(model, tree_num=tree_num,
                               feature_names=[mdl.get_feature_names(model)[i] for i in model.estimators_features_[tree_num]])
        """

        # get base model
        if re.search('RandomForest|ExtraTrees|Bagging|AdaBoost', str(model)):
            # warning Bagging and Adaboost base_estimator should be DecisionTree
            model = model.estimators_[tree_num]
        elif re.search('GradientBoostingClassifier', str(model)):
            model = model.estimators_.ravel()[tree_num]
        elif re.search('DecisionTreeClassifier', str(model)):
            model = model
        else:
            raise ValueError('only support DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting')

        def recurse(features, left, right, child, lineage=[]):
            """
            Recurse leaf information

            :param features: feature name
            :param left:  left node
            :param right: right node
            :param child: node
            :param lineage:
            :return:
            """

            if child in left:
                parent = np.where(left == child)[0].item()
                split = '<='
            else:
                parent = np.where(right == child)[0].item()
                split = '>'
            lineage.append(' '.join([features[parent], split, str(threshold[parent])]))

            if parent == 0:
                lineage.reverse()
                return lineage
            else:
                return recurse(features, left, right, parent, lineage)

        left_node = model.tree_.children_left
        right_node = model.tree_.children_right
        threshold = model.tree_.threshold
        features = [feature_names[i] for i in model.tree_.feature]
        idx = np.argwhere(left_node == -1)[:, 0]

        # show leaf nodes
        out = {}
        for leaf in idx:
            out.update({f'node_{leaf}': recurse(features, left_node, right_node, leaf, lineage=[])})
        return out

    @classmethod
    def dump_xgboost(cls, model, tree_num):
        """
        Show rules from xgboost model

        :param model: model
        :param tree_num: tree_num in xgboost
        :return:

        Examples:
            dump_xgboost(model, tree_num=10)
        """

        def recurse(trees_2df, child, lineage=[]):
            """
            Recurse child information

            :param trees_2df:  trees_to_dataframe
            :param child: node
            :param lineage:
            :return:
            """

            parent_code = dict(
                zip([int(x.split('-')[1]) for x in trees_2df['Yes']] + [int(x.split('-')[1]) for x in trees_2df['No']],
                    [int(x.split('-')[1]) for x in trees_2df['ID']] * 2))
            trees_2df = trees_2df.to_dict('index')

            while child > 0:
                idx1 = child % 2 == 1
                idx2 = re.search('-' + str(child) + '$', trees_2df[parent_code[child]]['Missing']) is not None
                if (idx1) & (idx2):
                    lineage.append(
                        f'{trees_2df[parent_code[child]]["Feature"]}<{trees_2df[parent_code[child]]["Split"]}|Missing')
                elif (idx1) & (~idx2):
                    lineage.append(
                        f'{trees_2df[parent_code[child]]["Feature"]}<{trees_2df[parent_code[child]]["Split"]}')
                elif (~idx1) & (idx2):
                    lineage.append(
                        f'{trees_2df[parent_code[child]]["Feature"]}>={trees_2df[parent_code[child]]["Split"]}|Missing')
                else:
                    lineage.append(
                        f'{trees_2df[parent_code[child]]["Feature"]}>={trees_2df[parent_code[child]]["Split"]}')
                child = parent_code[child]

            lineage.reverse()
            return lineage

        trees_2df = model.trees_to_dataframe()
        trees_2df = trees_2df[trees_2df['Tree'] == tree_num]
        idx = (pd.isnull(trees_2df['Yes'])) & (pd.isnull(trees_2df['No']))
        child_node = trees_2df[idx]['Node'].tolist()
        trees_2df = trees_2df[~idx].reset_index(drop=True).set_index('Node')

        # show leaf nodes
        out = {}
        for leaf in child_node:
            out.update({f'node_{leaf}': recurse(trees_2df, leaf, lineage=[])})
        return out

    @classmethod
    def dump_lightgbm(cls, model, tree_num):
        """
        Show rules from lightgbm model

        :param model: model
        :param tree_num: tree_num in lightgbm
        :return:

        Examples:
            dump_lightgbm(model, tree_num=10)
        """

        def recurse(trees_2df, child, lineage=[]):
            """
            Recurse child information

            :param trees_2df:  trees_to_dataframe
            :param child: node
            :param lineage:
            :return:
            """

            idx = pd.isnull(trees_2df['split_feature'])
            left_child = trees_2df[~idx]['left_child'].tolist()
            right_child = trees_2df[~idx]['right_child'].tolist()
            decision_type = {'>=': '<', '<=': '>', '==': '!=', '>': '<=', '<': '>='}

            while True:
                if child in left_child:
                    idx = trees_2df['left_child'] == child
                    if trees_2df[idx]['missing_direction'].values[0] == 'left':
                        lineage.append(
                            f'{trees_2df[idx]["split_feature"].values[0]}{trees_2df[idx]["decision_type"].values[0]}{trees_2df[idx]["threshold"].values[0]}|Missing')
                    else:
                        lineage.append(
                            f'{trees_2df[idx]["split_feature"].values[0]}{trees_2df[idx]["decision_type"].values[0]}{trees_2df[idx]["threshold"].values[0]}')

                elif child in right_child:
                    idx = trees_2df['right_child'] == child
                    if trees_2df[idx]['missing_direction'].values[0] == 'right':
                        lineage.append(
                            f'{trees_2df[idx]["split_feature"].values[0]}{decision_type[trees_2df[idx]["decision_type"].values[0]]}{trees_2df[idx]["threshold"].values[0]}|Missing')
                    else:
                        lineage.append(
                            f'{trees_2df[idx]["split_feature"].values[0]}{decision_type[trees_2df[idx]["decision_type"].values[0]]}{trees_2df[idx]["threshold"].values[0]}')

                else:
                    lineage.reverse()
                    return lineage

                child = trees_2df[idx]['node_index'].values[0]

        trees_2df = model.trees_to_dataframe()
        trees_2df = trees_2df[trees_2df['tree_index'] == tree_num].reset_index(drop=True)

        # show leaf nodes
        out = {}
        for leaf in trees_2df[pd.isnull(trees_2df['split_feature'])]['node_index']:
            out.update({f'node_{int(leaf.split("-L")[1])}': recurse(trees_2df, leaf, [])})
        return out

    @classmethod
    def plot_sklearn_tree(cls, model, tree_num, feature_names, save_path=None):
        """
        Plot tree rules from sklearn tree model

        :param model: model
        :param tree_num: tree_num in RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting
        :param feature_names:  feature_names
        :param save_path:  pic save_path
        :return:

        Examples:
            (1) plot_sklearn_tree(model, tree_num=0, feature_names=mdl.get_feature_names(model),save_path=None)
            (2) plot_sklearn_tree(model, tree_num=90, feature_names=mdl.get_feature_names(model), save_path='E:/huarui/')
            (3) # bagging
                tree_num =10
                plot_sklearn_tree(model, tree_num=tree_num,
                                  feature_names=[mdl.get_feature_names(model)[i] for i in model.estimators_features_[tree_num]],
                                  save_path=None)
        """

        # get base model
        if re.search('RandomForest|ExtraTrees|Bagging|AdaBoost', str(model)):
            # warning Bagging and Adaboost base_estimator should be DecisionTree
            model = model.estimators_[tree_num]
        elif re.search('GradientBoostingClassifier', str(model)):
            model = model.estimators_.ravel()[tree_num]
        elif re.search('DecisionTreeClassifier', str(model)):
            model = model
        else:
            raise ValueError('only support DecisionTree/RandomForest/ExtraTrees/Bagging/AdaBoost/GradientBoosting')

        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data, node_ids=True, feature_names=feature_names,
                        filled=False, rounded=True, special_characters=False)
        pic = Source(dot_data.getvalue())

        # save
        if save_path is not None:
            pic.render(filename=join(save_path, f'tree{tree_num}'), cleanup=True, format='png')
        return pic

    @classmethod
    def plot_xgboost_tree(cls, model, tree_num, save_path=None):
        """
        Plot tree rules from xgboost tree

        :param model: model
        :param tree_num: tree_num in xgboost
        :param save_path:  pic save_path
        :return:

        Examples:
            plot_xgboost_tree(model, tree_num=10)
        """

        trees = model.get_dump('', with_stats=False, dump_format='dot')
        max_depth = model.params['max_depth']
        str_ = trees[tree_num]

        for i in range(2 ** max_depth - 1):
            for j in [1, 2]:
                find1 = str_.find(f'-> {2 * i + j}')
                if find1 != -1:
                    find2 = find1 + str_[find1:].find(']\n')
                    med = str_[find1:find2]
                    if j == 1:  # 走左边永远通过
                        med = med.replace('#FF0000', '#0000FF').replace('yes, missing', ' pass&missing').replace('no',
                                                                                                                 'pass')
                    else:  # 走右边永远拒绝
                        med = med.replace('#0000FF', '#FF0000').replace('yes, missing', ' reject&missing').replace('no',
                                                                                                                   'reject')
                    str_ = str_[:find1] + med + str_[find2:]

        for i in range(2 ** (max_depth + 1) - 1):
            find1 = str_.find(f']\n\n    {i} [ label="leaf=')
            if find1 != -1:
                find2 = find1 + str_[find1:].find('" ]\n')
                str_ = str_[:find1] + str_[find1:find2] + f'\n #node={i}' + str_[find2:]
        pic = Source(str_)

        # save
        if save_path is not None:
            pic.render(filename=join(save_path, f'tree{tree_num}'), cleanup=True, format='png')
        return pic

    @classmethod
    def plot_lightgbm_tree(cls, model, tree_num, save_path=None):
        """
        Plot tree rules from lightgbm tree

        :param model: model
        :param tree_num: tree_num in lightgbm
        :param save_path:  pic save_path
        :return:

        Examples:
            plot_lightgbm_tree(model, tree_num=10)
        """

        pic = lgb.create_tree_digraph(model, tree_index=tree_num, precision=3, orientation='vertical')

        # save
        if save_path is not None:
            pic.render(filename=join(save_path, f'tree{tree_num}'), cleanup=True, format='png')
        return pic

    # plot catboost tree: model.plot_tree(tree_num, pool=model.dtrain)

    @classmethod
    def dumptree(cls, model, tree_num, type_, feature_names, save_path):
        """
        Plot or show tree rules from tree models

        :param model: model
        :param tree_num: tree_num
        :param type_: the presentation of the rules support all/pic/txt
        :param feature_names: feature_names in model
        :param save_path: pic save_path
        :return:

        Examples:
            (1) rule, pic = dumptree(model, tree_num=0, type_='all', feature_names=mdl.get_feature_names(model), save_path=None)
            (2) rule = dumptree(model, tree_num=0, type_='txt', feature_names=mdl.get_feature_names(model), save_path=None)
            (3) pic = dumptree(model, tree_num=0, type_='pic', feature_names=mdl.get_feature_names(model), save_path='E:/huarui/')
            (4) # bagging
                tree_num = 10
                rule, pic = dumptree(model, tree_num=tree_num, type_='all',
                                     feature_names=[mdl.get_feature_names(model)[i] for i in model.estimators_features_[tree_num]],
                                     save_path=None)
        """

        if re.search('xgboost', str(model)):
            txt = cls.dump_xgboost(model, tree_num)
            pic = cls.plot_xgboost_tree(model, tree_num, save_path)

        elif re.search('lightgbm', str(model)):
            txt = cls.dump_lightgbm(model, tree_num)
            pic = cls.plot_lightgbm_tree(model, tree_num, save_path)

        else:
            txt = cls.dump_sklearn_tree(model, tree_num, feature_names)
            pic = cls.plot_sklearn_tree(model, tree_num, feature_names, save_path)

        if type_ == 'all':
            return txt, pic
        elif type_ == 'pic':
            return pic
        elif type_ == 'txt':
            return txt
        else:
            raise ValueError('type_ only support all/pic/txt')


def model_kfold(n_splits, df, feas, target, model_type, importance_type=None, params=None):
    """
    Kfold trainning samples to view features and model performance

    :param n_splits: split number
    :param df: df
    :param feas: feas
    :param target: target
    :param model_type: model_type
    :param params: importance_type to drop feas
    :param params: params in model
    :return:

    Examples:
        (1) model_dict, model_performance = model_kfold(n_splits=5, df=x_train, feas=feas,
                                                target=target, model_type='lightgbm', params=None)
        (2) model_dict, model_performance = model_kfold(n_splits=5, df=x_train, feas=feas,
                                                target=target, model_type='xgboost',
                                                importance_type='total_gain', params=None)
        (3) model_dict, model_performance = model_kfold(n_splits=5, df=x_train, feas=feas,
                                                target=target, model_type='LogisticRegression',
                                                importance_type='pval', params=None)
    """

    df = df.reset_index()
    train_x = df[feas + [target]].copy()
    train_y = df[target].copy()

    # split train_x,train_y
    model_dict, ks_train, auc_train, ks_validate, auc_validate = {}, [], [], [], []
    kf = GroupKFold(n_splits=n_splits).split(train_x, train_y, groups=df['index'])
    for i, (train_fold, validate) in enumerate(kf):
        X_train, X_validate = train_x.loc[train_fold].reset_index(drop=True), train_x.loc[validate].reset_index(
            drop=True)
        print('*' * 60 + ' item {} '.format(i) + '*' * 60)
        print(f'X_train shape is {X_train.shape} target mean is {round(X_train[target].mean(), 6)}, '
              f'X_validate shape is {X_validate.shape} target mean is {round(X_validate[target].mean(), 6)}')

        # fit
        model = ModelApply.model_fit(model_type, X_train[feas], X_train[target], params=params)
        if importance_type is not None:
            if re.search('statsmodels', str(model)):
                model, _ = ModelApply.stepwise_lr(X_train, target, model, params=params, drop_cnt=0, by='pval', step=1,
                                                  thresh_pval=0.05)
            else:
                model, _ = ModelApply.stepwise_tree(X_train, None, target, model, params=params, drop_cnt=0,
                                                    by=importance_type, step=10, thresh_imp=0)

        # performance
        X_train_ks, X_train_auc, _ = evaluate_performance(X_train[target],
                                                          ModelApply.model_predict(X_train[get_feature_names(model)],
                                                                                   model), to_plot=False)
        X_validate_ks, X_validate_auc, _ = evaluate_performance(X_validate[target],
                                                                ModelApply.model_predict(
                                                                    X_validate[get_feature_names(model)],
                                                                    model), to_plot=False)
        model_dict.update({i: model})
        ks_train.append(X_train_ks)
        auc_train.append(X_train_auc)
        ks_validate.append(X_validate_ks)
        auc_validate.append(X_validate_auc)

    # out
    df.drop(columns='index', axis=1, errors='ignore', inplace=True)
    model_performance = pd.DataFrame({'X_train_ks': ks_train, 'X_train_auc': auc_train,
                                      'X_validate_ks': ks_validate, 'X_validate_auc': auc_validate})
    model_performance['ks_var'] = np.var(model_performance[['X_train_ks', 'X_validate_ks']], axis=1)
    model_performance['auc_var'] = np.var(model_performance[['X_train_auc', 'X_validate_auc']], axis=1)

    return model_dict, model_performance

    # use func: model_kfold return value model_dict to get kfold model performance
    # out = {}
    # data_dict = {'x_train': x_train, 'x_test': x_test, 'oot': oot}
    #
    # for index, model in model_dict.items():
    #     per_perf = []
    #     for data in data_dict.values():
    #         ks_, auc_, _ = mdl.evaluate_performance(data[target],
    #                                                 mdl.ModelApply.model_predict(data[mdl.get_feature_names(model)],
    #                                                                              model), to_plot=False)
    #         per_perf = per_perf + [ks_, auc_]
    #     out.update({index: per_perf})
    #
    # out = pd.DataFrame(out, index=[f'{c}_{perf}' for c in data_dict.keys() for perf in ['ks', 'auc']]).T
    # out['ks_var'] = np.var(out[[c for c in out if re.search('_ks$', c)]], axis=1)
    # out['auc_var'] = np.var(out[[c for c in out if re.search('_auc$', c)]], axis=1)

    # use func model_kfold return value model_dict to get kfold model importance
    # importance_type = 'gain'
    # for index, model in model_dict.items():
    #     feas_imp_ = mdl.ModelImportance.tree_importance(model, importance_type=importance_type)
    #     feas_imp_.columns = [f'{c}_{index}' for c in feas_imp_.columns]
    #     if index == 0:
    #         feas_imp = feas_imp_
    #     else:
    #         feas_imp = feas_imp.merge(feas_imp_, left_index=True, right_index=True, how='outer')
    #
    # index = [0, 1, 4]
    # feas_imp['importance_all'] = feas_imp[
    #     [c for c in feas_imp if re.search('|'.join(set(['gain_' + str(c) for c in index])), c)]].sum(axis=1)