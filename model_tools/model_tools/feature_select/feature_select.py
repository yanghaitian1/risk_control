import json
from os.path import join

import numpy as np
import pandas as pd
import statsmodels.api as sm
from minepy import MINE
from scipy.stats import chi2_contingency
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from ..gentools.gentools import keep_df_value
from ..gentools.advtools import cut_feas, GetInformGain, WOE, JsonEncoder


def drop_na_cols(df, thresh=0.9, cols=None, type_='calc', is_flag=True, exclude=None):
    """
    Remove features which null value is greater than the threshold and add flag

    :param df: df
    :param thresh: null value thresh
    :param cols: cols to calc null value or drop directly
    :param type_: calc/no_cala if calc: need to compare cols null value and thresh. if no_calc: drop directly
    :param is_flag: whether to add flag
    :param exclude: features do not need to calc null value
    :return:

    Examples:
        (1) x_train, _ = drop_na_cols(x_train, thresh=0.9, cols=x_train.select_dtypes(np.number).columns, type_='calc', is_flag=True, exclude=[])
        (2) x_train, _ = drop_na_cols(x_train, thresh=0.9, cols=None, type_='calc', is_flag=False, exclude=[])
        (3) x_test, _ = drop_na_cols(x_test, thresh=0.9, cols=['ord_amt_sum_p36d','morn_ord_succ_amt_min_p7d'],
                      type_='nocalc', is_flag=True, exclude=[])
    """

    if exclude is None:
        exclude = []
    if cols is None:
        cols = list(df)
    cols = list(set(cols) - set(exclude))

    # calc null value
    if type_ == 'calc':
        stats = df[cols].isnull().sum() / df.shape[0]
        cols_drop = stats.loc[(stats > thresh)].index.tolist()
        print('# of columns to drop: {}'.format(len(cols_drop)))
    else:
        cols_drop = cols

    # add flag
    if is_flag is True:
        print('# start add is_na flag')
        for feas in tqdm(cols_drop):
            df[feas + '_isna'] = df[feas].apply(lambda x: 1 if pd.isnull(x) else 0)

    return df.drop(cols_drop, axis=1, errors='ignore'), cols_drop


def drop_singleValue_cols(df, thresh=0.9, cols=None, type_='calc', is_flag=True, exclude=None, save_file=None):
    """
    Remove features which single value is greater than the threshold and add flag

    :param df: df
    :param thresh: single value thresh
    :param cols: list/dict cols to calc single value or drop directly
    :param type_: calc/no_cala if calc: need to compare cols single value and thresh. if no_calc: drop directly
    :param is_flag: whether to add flag
    :param exclude: features do not need to calc single value
    :param save_file: file to save single information
    :return:

    Examples:
        (1) x_train, _ = drop_singleValue_cols(x_train, thresh=0.9, cols=None, type_='calc', is_flag=True, exclude=None, save_file=None)
        (2) x_test, _ = drop_singleValue_cols(x_test, thresh=0.9, cols={'ovd_days_rf_repay_min_p36d':0,'ovd_days_rf_due_mean_p96d':0},
                               type_='nocalc', is_flag=True, exclude=None,
                               save_file='E:/huarui/singlevalue_cfg')
    """

    if exclude is None:
        exclude = []
    if cols is None:
        cols = list(df)

    if (isinstance(cols, list)) & (type_ == 'calc'):
        thresh_cnt = thresh * (df.shape[0])
        cfg, cols = {}, list(set(cols) - set(exclude))
        for fea in tqdm(cols):
            a = np.array(df[fea])
            try:
                a = a[~np.isnan(a)]
            except TypeError:
                a = a[~pd.isnull(a)]
            value = pd.value_counts(a)
            if value.values[0] > thresh_cnt:
                cfg.update({fea: value.index[0]})

    elif (isinstance(cols, dict)) & (type_ == 'nocalc'):
        cfg = cols
        _ = [cfg.pop(k) for k in list(set(cfg.keys()).intersection(set(exclude)))]

    else:
        raise ValueError(f'cols and type_ should correspond one-to-one')

    if is_flag is True:
        print('# start add is_mode flag')
        for fea, values in tqdm(cfg.items()):
            fea_flag = '_add_mode_flag'
            df[fea + fea_flag] = 0
            df.loc[df[fea] == values, fea + fea_flag] = 1

    if save_file is not None:
        with open(save_file, 'w') as json_file:
            json.dump(cfg, json_file, ensure_ascii=False, cls=JsonEncoder)

    return df.drop(list(cfg.keys()), axis=1, errors='ignore'), cfg


class FilterIndex(object):
    """
    select features by Filter
    """

    def __init__(self, x, y):
        """
        :param x: input features
        :param y: target y
        """
        self.x = x
        self.y = y

    def CalcPerson(self):
        """
        Calc person

        :return: {'Person': Person}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        corr = pearsonr(a[0, :], a[1, :])
        return {'Person': corr[0]}

    def CalcFclassif(self):
        """
        Calc Fclassif

        :return: {'Fclassif': Fclassif}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        score, p_value = f_classif(a[0, :].reshape(-1, 1), a[1, :])
        return {'Fclassif': score[0]}  # p_value[0]

    def CalcChi2(self):
        """
        Calc Chi2Score

        :return: {'Chi2': Chi2}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        value = chi2_contingency(np.array(pd.crosstab(a[0, :], a[1, :])), correction=False)
        return {'Chi2': value[0]}  # value[1]

    def CalcInformGain(self):
        """
        Calc Entropy

        :return: {'inform_gain': inform_gain, 'inform_gain_rto': inform_gain_rto}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        value = GetInformGain().get_information_gain(a[0, :], a[1, :])
        return {'inform_gain': value[0], 'inform_gain_rto': value[1]}

    def CalcIV(self):
        """
        Calc Iv

        :return: {'IV': IV, 'absWOE': absWOE}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass
        value = WOE().woe_single_x(a[0, :], a[1, :], 1)
        return {'IV': value[1], 'absWOE': value[2]}

    def CalcAucKs(self):
        """
        Calc Auc&Ks

        :return: {'AUC': AUC, 'KS': KS}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass

        try:
            lm = sm.Logit(a[1, :], sm.add_constant(a[0, :], prepend=False, has_constant='add')).fit(disp=0)
            predictions = lm.predict(sm.add_constant(a[0, :], prepend=False, has_constant='add'))
            fpr, tpr, thresholds = roc_curve(a[1, :], predictions)
            roc_auc = round(auc(fpr, tpr), 4)
            ks = max(tpr - fpr)
            return {'AUC': roc_auc, 'KS': ks}
        except:
            return {'AUC': np.nan, 'KS': np.nan}

    def CalcMic(self, alpha=0.6, c=15):
        """
        calc Mine
        https://blog.csdn.net/qq_27586341/article/details/90603140
        http://minepy.sourceforge.net/docs/1.0.0/python.html

        :return: {'MIC': MIC}
        """

        a = np.array([self.x, self.y])
        try:
            a = a[:, ~np.isnan(a).any(axis=0)]
        except TypeError:
            pass

        mine = MINE(alpha=alpha, c=c)
        mine.compute_score(a[0, :], a[1, :])
        return {'MIC': mine.mic()}

    def CalcFilterIndex(self, calc_list):
        """
        Calc specified evaluation index of the feature

        :param calc_list: support ["all"] or one or more in ['CalcPerson', 'CalcFclassif', 'CalcChi2', 'CalcInformGain', 'CalcIV', 'CalcAucKs', 'CalcMic']
        :return: dict, {calc:value}

        Examples:
            (1) FilterSelect(x_train['ovd_days_rf_repay_last_p186d'], x_train['p1_0']).CalcFilterIndex('all')
            (2) FilterSelect(x_train['ovd_days_rf_repay_last_p186d'], x_train['p1_0']).CalcFilterIndex(['CalcPerson','CalcFclassif','CalcInformGain'])
            (3) FilterSelect(x_train['ovd_days_rf_repay_last_p186d'], x_train['p1_0']).CalcFilterIndex('CalcPerson')
        """

        if isinstance(calc_list, str):
            calc_list = [calc_list]
        if calc_list == ['all']:
            calc_list = ['CalcPerson', 'CalcFclassif', 'CalcChi2', 'CalcInformGain',
                         'CalcIV', 'CalcAucKs', 'CalcMic']
        calc_list = ['self.' + c for c in calc_list]

        out_dict = {}
        for calc in calc_list:
            value = eval(calc)()
            out_dict.update(value)

        return out_dict


def calc_feas_filterindex(data, feas_list, target, calc_list, cut,
                          drop_value, exclude_feas=None, save_path=None, **kwargs):
    """
    Calc feas filter index prepare for feature select

    :param data: dataframe
    :param feas_list: feature_list to calc index
    :param target: str,target for calc index
    :param calc_list: support ["all"] or one or more in ['CalcPerson', 'CalcFclassif', 'CalcChi2', 'CalcInformGain', 'CalcIV', 'CalcAucKs', 'CalcMic']
    :param cut: support None/int/list/dict
    :param drop_value: the value that needs to be deleted in the feature
    :param exclude_feas: features do not need to calc filter index
    :param kwargs: thred/max_bins
    :return: dataframe with filter index for every feature

    Examples:
        (1) feas_list = ['ely_gt1_repay_amt_rf_repay_mean_p36d','ely_gt1_repay_amt_rf_repay_mean_p66d','ovd_days_rf_repay_last_p186d']
            filterindex,_ = fs.calc_feas_filterindex(x_train, feas_list, target='p1_0', calc_list='all',
                                 cut={'max_depth': 3, 'min_samples_leaf': 0.03},
                                 drop_value=None, exclude_feas=None, save_path='E:/git/yht_tools_new/', thred=2)

        (2) feas_list = ['OBJECTNO','ely_gt1_repay_amt_rf_repay_mean_p36d','ely_gt1_repay_amt_rf_repay_mean_p66d','ovd_days_rf_repay_last_p186d']
            filterindex,_ = fs.calc_feas_filterindex(x_train, feas_list, target='p1_0', calc_list='all',
                                 cut={'max_depth': 3, 'min_samples_leaf': 0.03},
                                 drop_value=None, exclude_feas=['OBJECTNO'], max_bins=5)

        (3) filterindex,_ = fs.calc_feas_filterindex(x_train, feas_list=x_train.select_dtypes(np.number).columns,
                                 target='p1_0', calc_list='CalcPerson', cut=None, drop_value=None,
                                 exclude_feas=['OBJECTNO','target','PUTOUTSERIALNO','CUSTOMERID','order_create_time',
                                               'order_create_at','SERIALNO','NATURALOCCURDATE','APPLSERIALNO','ruleSERIALNO',
                                               'RULESERIALNO','JIAOYIRULE_SERIALNO','ref_time'])

        filter_index= pd.read_csv('E:/huarui/filter_index.csv',index_col=0)
    """

    inform_dict, feas_cntcut = {}, []
    if exclude_feas is None:
        exclude_feas = []
    feas_list = list(set(feas_list) - set(exclude_feas))

    # start calc filter index
    for fea in tqdm(feas_list):
        df = data[[fea] + [target]]
        # drop value
        df = keep_df_value(df, fea, drop_value)[1]
        df[fea] = df[fea].astype(float)

        # get cut bins
        if cut is not None:
            bins = cut_feas(df, fea, target, cut_params=cut,
                            thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            if (bins == [-np.inf, np.inf]) & (isinstance(cut, int)):
                bins = cut_feas(df, fea, target, cut_params={'max_depth': 2, 'min_samples_leaf': 0.02},
                                thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            if bins != [-np.inf, np.inf]:
                df[fea] = pd.cut(df[fea], bins, labels=range(len(bins) - 1))
                df[fea] = df[fea].astype(float)
            else:
                feas_cntcut.append(fea)
                continue

        value = FilterIndex(df[fea], df[target]).CalcFilterIndex(calc_list)
        inform_dict.update({fea: value})

    print('# of feas cnt not cut: {}'.format(len(feas_cntcut)))
    feas_index = pd.DataFrame.from_dict(inform_dict).T

    # save
    if save_path is not None:
        feas_index.to_csv(join(save_path, 'filterindex.csv'))

    return feas_index, feas_cntcut


def drop_cols_rf_corr(feas_imp, corr_table, thresh):
    """
    When the feature correlation is high, delete the features with low importance

    :param feas_imp: feas_imp
    :param corr_table: feas corr_table
    :param thresh: corr thresh
    :return:

    Examples:
        # xgboost model
        feas_imp = mdl.ModelImportance.tree_importance(model, 'total_gain')
        corr_table = x_train[feas].corr()
        drop_df = drop_cols_rf_corr(feas_imp, corr_table, thresh=0.6)

        # check
        # idx = 0
        # print(f'drop_col is {drop_df["drop_cols"][idx]}')
        # cols = drop_df['compare_cols'][idx].split('+')
        # corr_table[corr_table.index.isin(cols)][cols].merge(feas_imp[feas_imp.index.isin(cols)], left_index=True, right_index=True)
    """

    corr_table = abs(corr_table)
    drop_cols, compare_cols, corrs = [], [], []
    corr_value = corr_table[corr_table < 1].max().max()

    while corr_value > thresh:
        tmp = pd.DataFrame(corr_table[corr_table == corr_value].max()).dropna()
        i = tmp.index.tolist()[0]
        j = tmp.index.tolist()[1]
        compare_cols.append('{}+{}'.format(i, j))
        corrs.append(corr_value)
        if feas_imp[feas_imp.index == i].values[0][0] > feas_imp[feas_imp.index == j].values[0][0]:
            drop_col = j
        else:
            drop_col = i

        # update corr_table
        drop_cols.append(drop_col)
        idx = (corr_table.index != drop_col)
        corr_table = corr_table[idx].drop(columns=drop_col)
        corr_value = corr_table[corr_table < 1].max().max()

    return pd.DataFrame({'compare_cols': compare_cols, 'corr': corrs, 'drop_cols': drop_cols})
