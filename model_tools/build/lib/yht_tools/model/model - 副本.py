import math
import traceback
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import xgboost as xgb
from scipy.stats import chi2_contingency
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from ..ytools.ytools import GetInformGain, WOE, keep_df_value, cut_feas


class FeatureSelect(object):
    def __init__(self, x, y):
        """
        x and y can be Series

        :param x: input features
        :param y: target y
        """
        self.x = x
        self.y = y

    def FclassifSelect(self):
        """
        get FScore
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        score, p_value = f_classif(a[0, :].reshape(-1, 1), a[1, :])
        return score[0], p_value[0]

    def Chi2Select(self):
        """
        get Chi2Score
        """
        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        value = chi2_contingency(np.array(pd.crosstab(a[0, :], a[1, :])), correction=False)

        return value[0], value[1]

    def InformGainSelect(self):
        """
        get Entropy
        """
        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass

        return GetInformGain().get_information_gain(self.x, self.y)

    def PersonSelect(self):
        """
        get Person
        """
        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        corr = pearsonr(a[0, :], a[1, :])

        return [corr[0]]

    def IVSelect(self):
        """
        get iv
        """
        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        value = WOE().woe_single_x(a[0, :], a[1, :], 1)

        return [value[1]]

    def AucSelect(self):
        """
        get each auc_ks
        """
        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass

        lm = sm.Logit(a[1, :], sm.add_constant(a[0, :], prepend=False, has_constant='add')).fit(disp=0)
        predictions = lm.predict(sm.add_constant(a[0, :], prepend=False, has_constant='add'))
        fpr, tpr, thresholds = roc_curve(a[1, :], predictions)
        roc_auc = round(auc(fpr, tpr), 4)
        ks = max(tpr - fpr)

        return roc_auc, ks


def select_feas_by_filter(df, cols, target, func, key=None,
                          percentile=100, drop_value=None, cut=None, exclude=[], save_path=None):
    func_dict = {'FclassifSelect': ['Fscore', 'Pvalue'],
                 'Chi2Select': ['Chi2score', 'Pvalue'],
                 'InformGainSelect': ['Inform_gain', 'Inform_gain_rto'],
                 'PersonSelect': ['Pearsonr'],
                 'AucSelect': ['AUC', 'KS'],
                 'IVSelect': ['IV']}

    value_list = []
    feas_list = []
    rename_columns = func_dict[func]

    for feas in tqdm(list(set(cols) - set(exclude))):
        data = df[[feas] + [target]]
        data = data.dropna(subset=[feas]).reset_index(drop=True)

        ################
        # keep_df_value
        ################
        data = keep_df_value(data, feas, drop_value)

        ################
        # cut_feas
        ################
        if (isinstance(cut, int)):
            data = cut_feas(data, feas, target, cut)[0]

        ################
        # select feas by given func
        ################

        if func == 'FclassifSelect':
            value = FeatureSelect(data[feas], data[target]).FclassifSelect()
        elif func == 'Chi2Select':
            value = FeatureSelect(data[feas], data[target]).Chi2Select()
        elif func == 'InformGainSelect':
            value = FeatureSelect(data[feas], data[target]).InformGainSelect()
        elif func == 'PersonSelect':
            value = FeatureSelect(data[feas], data[target]).PersonSelect()
        elif func == 'AucSelect':
            value = FeatureSelect(data[feas], data[target]).AucSelect()
        elif func == 'IVSelect':
            value = FeatureSelect(data[feas], data[target]).IVSelect()
        else:
            raise ValueError('Unknown func')

        value_list.append(value)
        feas_list.append(feas)

    df_score = pd.DataFrame({'feas': feas_list, 'value': value_list})
    df_score = df_score.merge(pd.DataFrame([[df_score['value'][i][j]
                                             for j in range(len(df_score['value'][0]))]
                                            for i in range(len(df_score))], columns=rename_columns),
                              left_index=True, right_index=True)

    df_score = df_score.sort_values(rename_columns[0], ascending=False, key=key).reset_index(drop=True)

    ##########################
    # drop not important feas
    ##########################
    # add Support
    idx = df_score.index <= int(df_score.shape[0] * percentile / 100)
    df_score.loc[idx, 'Support'] = True
    df_score.loc[~idx, 'Support'] = False

    # drop
    df = df.drop(df_score[df_score['Support'] == False]['feas'].tolist(), axis=1, errors='ignore')

    # save df_score
    if save_path is not None:
        df_score.to_csv(join(save_path, '{}_select'.format(func)), index=False)

    return df, df_score


def xgb_apply(model, X_train, dummy=True):
    """
    params model: booster model
    params dummy
    """

    try:
        xgb_apply_feas = pd.DataFrame(model.apply(X_train))
        xgb_apply_feas.columns = ['tree{}'.format(c) for c in range(xgb_apply_feas.shape[1])]
        for f in list(xgb_apply_feas):
            xgb_apply_feas[f] = xgb_apply_feas[f].astype(str)

        if dummy is False:
            return xgb_apply_feas

        else:
            need = model.get_booster().trees_to_dataframe()
            idx = pd.isnull(need['Yes'])
            need = need[~idx].reset_index(drop=True)
            xgb_apply_feas = pd.get_dummies(xgb_apply_feas)

            def split_link(x, x_list):
                if x >= 0:
                    x_list.append(x)
                    return split_link(math.ceil(x / 2) - 1, x_list)
                else:
                    return sorted(x_list)[:-1]

            tree_split_link = {}
            for c in range(2 ** (model.max_depth + 1) - 2):
                tree_split_link.update({c + 1: split_link(c + 1, [])})

            new_cols_list = []
            old_cols_list = []
            for tree in range(model.n_estimators):
                tmp = need[need['Tree'] == tree]
                for value in range(2 ** (model.max_depth + 1) - 2):
                    new_cols_list.append(f'tree{tree}_node_{value + 1}_' + \
                                         '|'.join(
                                             list(set(tmp[tmp['Node'].isin(tree_split_link[value + 1])]['Feature']))))
                    old_cols_list.append(f'tree{tree}_{value + 1}')

            xgb_apply_feas.rename(columns=dict(zip(old_cols_list, new_cols_list)), inplace=True)
            return xgb_apply_feas

    except:
        err = traceback.format_exc()
        print(err)


class GetXGBImportance:
    def __init__(self):
        pass

    @classmethod
    def _get_boster_shapimp(cls, model, X_train, is_plot=True):

        """
        params model: XGB model
        params X_train: pd.DataFrame
        """

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        max_display = len(X_train)
        feature_names = list(X_train)
        global_shap_values = np.abs(shap_values).mean(0)

        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
        feature_inds = feature_order[:max_display]
        out = pd.DataFrame({'feature': [feature_names[i] for i in feature_inds], \
                            'shap_imp': global_shap_values[feature_inds]}).sort_values('shap_imp', ascending=False). \
            reset_index(drop=True)
        if is_plot is True:
            shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=out[out['shap_imp'] > 0].shape[0])

        return out

    @classmethod
    def _get_boster_imp(cls, model, X_train,
                        importance_type_list=['weight', 'gain', 'cover', 'total_gain', 'total_cover'],
                        ref_importance_type='total_gain', is_plot=True):

        data = pd.DataFrame(index=list(X_train))
        for importance_type in importance_type_list:
            try: # use xgb.XGBRegressor
                tmp = pd.DataFrame(model.get_booster().get_score(importance_type=importance_type), index=[0]). \
                    T.rename(columns={0: importance_type})
            except AttributeError:
                tmp = pd.DataFrame(model.get_score(importance_type=importance_type), index=[0]). \
                    T.rename(columns={0: importance_type})
            data = data.merge(tmp, left_index=True, right_index=True, how='left')
            data = data.fillna(0)

        data = data.sort_values(ref_importance_type, ascending=False).reset_index().rename(columns={'index': 'feature'})

        if is_plot is True:
            fig, ax = plt.subplots(figsize=(13, 13))
            xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=(data[ref_importance_type] > 0).sum(),
                                importance_type=ref_importance_type,
                                grid=False, show_values=False, xlabel=ref_importance_type)
            plt.show()
        return data

    @classmethod
    def get_boster_imp(cls, model, X_train):
        shapimp = cls._get_boster_shapimp(model, X_train, is_plot=False)
        imp = cls._get_boster_imp(model, X_train,
                              importance_type_list=['weight', 'gain', 'cover', 'total_gain', 'total_cover'],
                              ref_importance_type='total_gain', is_plot=False)
        imp = imp.merge(shapimp)
        return imp
