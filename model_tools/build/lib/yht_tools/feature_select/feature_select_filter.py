from os.path import join

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from ..plot.feature_plot import _plot_bivar
from ..ytools.ytools import GetInformGain, WOE, keep_df_value, cut_feas, make_dir


class FeatureSelect(object):
    def __init__(self, x, y):
        """
        x and y can be Series

        :param x: input features
        :param y: target y
        """
        self.x = x
        self.y = y

    def FclassifSelect(self, is_calc=True):
        """
        get FScore
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            score, p_value = f_classif(a[0, :].reshape(-1, 1), a[1, :])
            return score[0], p_value[0]
        else:
            return 0, 0

    def Chi2Select(self, is_calc=True):
        """
        get Chi2Score
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            value = chi2_contingency(np.array(pd.crosstab(a[0, :], a[1, :])), correction=False)
            return value[0], value[1]
        else:
            return 0, 0

    def InformGainSelect(self, is_calc=True):
        """
        get Entropy
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            return GetInformGain().get_information_gain(self.x, self.y)
        else:
            return 0, 0

    def PersonSelect(self, is_calc=True):
        """
        get Person
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            corr = pearsonr(a[0, :], a[1, :])
            return [corr[0]]
        else:
            return [0]

    def IVSelect(self, is_calc=True):
        """
        get iv
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            value = WOE().woe_single_x(a[0, :], a[1, :], 1)
            return [value[1]]
        else:
            return [0]

    def WOESelect(self, is_calc=True):
        """
        get iv
        """
        if is_calc:
            a = np.array([self.x, self.y])
            try:
                a = a[:, ~np.isnan(a).any(axis=0)]
            except TypeError:
                pass
            value = WOE().woe_single_x(a[0, :], a[1, :], 1)
            return [value[2]]
        else:
            return [0]

    def AucSelect(self, is_calc=True):
        """
        get each auc_ks
        """
        if is_calc:
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
        else:
            return 0, 0


def select_feas_by_filter(df, cols, target, func, key=None,
                          percentile=100, drop_value=None, cut=None, exclude=[],
                          save_path=None):
    func_dict = {'FclassifSelect': ['Fscore', 'Pvalue'],
                 'Chi2Select': ['Chi2score', 'Pvalue'],
                 'InformGainSelect': ['Inform_gain', 'Inform_gain_rto'],
                 'PersonSelect': ['Pearsonr'],
                 'AucSelect': ['AUC', 'KS'],
                 'IVSelect': ['IV'],
                 'WOESelect': ['WOE']}

    value_list = []
    feas_list = []
    rename_columns = func_dict[func]
    if save_path is None:
        file_save_path, pic_save_path = None, None
    else:
        file_save_path = save_path.get('file_save_path', None)
        pic_save_path = save_path.get('pic_save_path', None)
        make_dir(file_save_path)
        make_dir(pic_save_path)

    for feas in tqdm(list(set(cols) - set(exclude))):
        data = df[[feas] + [target]]
        # data = data.dropna(subset=[feas]).reset_index(drop=True)

        ################
        # keep_df_value
        ################
        data = keep_df_value(data, feas, drop_value)

        ################
        # cut_feas
        ################
        tmp_, st, __ = cut_feas(data, feas, target, cut)
        data = tmp_.copy()
        st['mean'] = data[target].mean()

        ###############
        # draw_pic
        ###############
        if pic_save_path is not None:
            pic = _plot_bivar(st, target, draw_lin=False, yaxis='count', title=feas,
                              pycharts=False, save_path=pic_save_path)

        ################
        # select feas by given func
        ################
        is_calc = True
        if cut == None:
            data[feas] = data[feas].astype(float)
        else:
            data = data[data[feas] != 'nan']
            if '(-inf, inf]' in data[feas].unique():
                is_calc = False
            data[feas] = data[feas].replace(dict(zip(data[feas].unique(), range(len(data[feas].unique())))))

        if func == 'FclassifSelect':
            value = FeatureSelect(data[feas], data[target]).FclassifSelect(is_calc)
        elif func == 'Chi2Select':
            value = FeatureSelect(data[feas], data[target]).Chi2Select(is_calc)
        elif func == 'InformGainSelect':
            data[feas] = data[feas].astype(str)
            value = FeatureSelect(data[feas], data[target]).InformGainSelect(is_calc)
        elif func == 'PersonSelect':
            value = FeatureSelect(data[feas], data[target]).PersonSelect(is_calc)
        elif func == 'AucSelect':
            try:
                value = FeatureSelect(data[feas], data[target]).AucSelect(is_calc)
            except:
                value = [1, 1]
        elif func == 'IVSelect':
            data[feas] = data[feas].astype(str)
            value = FeatureSelect(data[feas], data[target]).IVSelect(is_calc)
        elif func == 'WOESelect':
            data[feas] = data[feas].astype(str)
            value = FeatureSelect(data[feas], data[target]).WOESelect(is_calc)
        else:
            raise ValueError('Unknown func')

        value_list.append(value)
        feas_list.append(feas)

    df_score = pd.DataFrame({'feature': feas_list, 'value': value_list})
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
    df = df.drop(df_score[df_score['Support'] == False]['feature'].tolist(), axis=1, errors='ignore')

    # save df_score
    if file_save_path is not None:
        df_score.to_csv(join(file_save_path, '{}_select.csv'.format(func)), index=False)

    return df, df_score


def select_feas_all(df, cols, target,
                    drop_value=None, cut=None, exclude=[],
                    save_path=None):
    value_list = []
    feas_list = []

    if save_path is None:
        file_save_path, pic_save_path = None, None
    else:
        file_save_path = save_path.get('file_save_path', None)
        pic_save_path = save_path.get('pic_save_path', None)

    for feas in tqdm(list(set(cols) - set(exclude))):
        data = df[[feas] + [target]]
        # data = data.dropna(subset=[feas]).reset_index(drop=True)

        ################
        # keep_df_value
        ################
        data = keep_df_value(data, feas, drop_value)

        ################
        # cut_feas
        ################
        tmp_, st, __ = cut_feas(data, feas, target, cut)
        data = tmp_.copy()
        st['mean'] = data[target].mean()

        ###############
        # draw_pic
        ###############
        if pic_save_path is not None:
            pic = _plot_bivar(st, target, draw_lin=False, yaxis='count', title=feas,
                              pycharts=False, save_path=pic_save_path)

        ################
        # select feas by given func
        ################
        is_calc = True
        if cut == None:
            data[feas] = data[feas].astype(float)
            data = data.dropna(subset=[feas]).reset_index(drop=True)
        else:
            data = data[data[feas] != 'nan'].reset_index(drop=True)
            if '(-inf, inf]' in data[feas].unique():
                is_calc = False
            data[feas] = data[feas].replace(dict(zip(data[feas].unique(), range(len(data[feas].unique())))))

        Fclassify_ = FeatureSelect(data[feas], data[target]).FclassifSelect(is_calc)[0]
        Person_ = FeatureSelect(data[feas], data[target]).PersonSelect(is_calc)[0]
        try:
            auc_, ks_ = FeatureSelect(data[feas], data[target]).AucSelect(is_calc)
        except:
            auc_, ks_ = 1, 1

        data[feas] = data[feas].astype(str)
        InformGain_ = FeatureSelect(data[feas], data[target]).InformGainSelect(is_calc)[0]
        iv_ = FeatureSelect(data[feas], data[target]).IVSelect(is_calc)[0]

        value_list.append([Fclassify_, Person_, auc_, ks_, InformGain_, iv_])
        feas_list.append(feas)

    df_score = pd.DataFrame({'feature': feas_list, 'value': value_list})
    df_score['Fclassify_'] = [c[0] for c in df_score['value']]
    df_score['Person_'] = [c[1] for c in df_score['value']]
    df_score['auc_'] = [c[2] for c in df_score['value']]
    df_score['ks_'] = [c[3] for c in df_score['value']]
    df_score['InformGain_'] = [c[4] for c in df_score['value']]
    df_score['iv_'] = [c[5] for c in df_score['value']]
    df_score.drop(columns=['value'], axis=1, inplace=True)

    # save df_score
    if file_save_path is not None:
        df_score.to_csv(join(file_save_path, 'feas_sel.csv'), index=False)

    return df, df_score
