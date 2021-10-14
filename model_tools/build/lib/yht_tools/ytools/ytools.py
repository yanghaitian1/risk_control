import datetime as dt
import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from os.path import join

import at.model_fitting as mf
import numpy as np
import pandas as pd
from IPython import display
from gmssl import sm3, func
from impala.dbapi import connect
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def eval_value(value):
    '''
    get eval value
    '''
    if isinstance(value, int) or isinstance(value, float) \
            or isinstance(value, list) or isinstance(value, dict):
        return value

    elif isinstance(value, str):
        try:
            return eval(value)
        except:
            return value

    elif (value is None) or (pd.isnull(value)):
        return None

    else:
        raise ValueError(f'Unknown value={value}')


def drop_cols_rf_corr(df, cor_table, thre, metric):
    '''
    使用方法: tmp = drop_cols_rf_corr(df_imp,abs(cor_table),0.8,'importance')
    drop corr feas
    params: df should contain 'feature' and importance
    
    """
      feature	importance
      0	br_yixiang_m3_id_avg_monnum_proc	0.013071
      1	br_yixiang_m1_cell_nbank_allnum_proc	0.012303
    """
    
    '''

    df_imp = df.copy()
    drop_cols = []
    compare_cols = []
    corrs = []
    valu = cor_table[cor_table < 1].max().max()

    while valu > thre:
        tmp = pd.DataFrame(cor_table[cor_table == valu].max()).dropna()
        i = tmp.index.tolist()[0]
        j = tmp.index.tolist()[1]
        compare_cols.append('{}+{}'.format(i, j))
        corrs.append(valu)
        if df_imp[df_imp['feature'] == i][metric].tolist()[0] > df_imp[df_imp['feature'] == j][metric].tolist()[0]:
            drop_col = j
        else:
            drop_col = i

        # update cor_table
        drop_cols.append(drop_col)
        idx = (cor_table.index != drop_col)
        cor_table = cor_table[idx].drop(columns=drop_col)
        valu = cor_table[cor_table < 1].max().max()

    return pd.DataFrame({'compare_cols': compare_cols, 'corr': corrs, 'drop_cols': drop_cols})


def drop_na_cols(df, thresh=0.9, cols_drop=None, is_flag=True, exclude=[]):
    '''
    drop na & add flag
    '''

    if cols_drop is None:
        stats = df.isnull().sum() / df.shape[0]
        idx_ex = pd.Series(stats.index.isin(exclude), index=stats.index)
        stats_drop = stats.loc[(stats > thresh) & ~idx_ex]
        cols_drop = stats_drop.index.tolist()
        print('# of columns to drop: {}'.format(len(cols_drop)))

    else:
        stats_drop = cols_drop

    if is_flag is True:
        print('# start add is_na flag')
        for feas in tqdm(cols_drop):
            df[feas + '_isna'] = df[feas].apply(lambda x: 1 if pd.isnull(x) else 0)

    return df.drop(cols_drop, axis=1, errors='ignore'), stats_drop


def drop_singleValue_cols(df, thresh=0.9, Sigle_var=None, is_flag=True,
                          exclude=[], save_file=None):
    '''
    drop singlevalue & add flag
    '''

    if Sigle_var is None:
        Sigle_var = []
        for i in df:
            a = np.array(df[i])
            try:
                a = a[~np.isnan(a)]
            except TypeError:
                a = a[~pd.isna(a)]
            if list(pd.value_counts(a))[0] / len(a) > thresh:
                Sigle_var.append(i)

        Sigle_var = list(set(Sigle_var) - set(exclude))
        Sigle_var = dict(zip(Sigle_var, [np.nan] * len(Sigle_var)))
        print('# of columns to drop {}'.format(len(Sigle_var)))

    if is_flag is True:
        print('# start add is_mode flag')

        for feas, values in tqdm(Sigle_var.items()):
            if pd.isnull(values):
                mode = df[feas].mode().values[0]
                Sigle_var.update({feas: mode})
            else:
                mode = values
            try:
                feas_flag = '_is_mode_{}'.format(int(mode))
            except ValueError:
                feas_flag = '_is_mode_{}'.format(mode)

            df[feas + feas_flag] = 0
            idx = df[feas] == mode
            df.loc[idx, feas + feas_flag] = 1

    if save_file is not None:
        with open(save_file, 'w') as json_file:
            json.dump(Sigle_var, json_file, ensure_ascii=False, cls=JsonEncoder)

    return df.drop(list(Sigle_var.keys()), axis=1, errors='ignore'), Sigle_var


def keep_df_value(df, feas, drop_value=None):
    '''
    keep_values: some feas value need to drop first
    '''
    if isinstance(drop_value, int) or isinstance(drop_value, float):
        mask = df[feas] == drop_value
    elif isinstance(drop_value, list):
        mask = df[feas].isin(drop_value)
    elif callable(drop_value):
        mask = df[feas].apply(drop_value)
    else:
        mask = pd.Series([False] * len(df))

    return df[~mask].reset_index(drop=True)


def cut_feas_bykmean(data, feas, cut_params={'KMeans': 4}):
    """
    cut feas by kmean
    """
    df = data[[feas]].sort_values(feas).reset_index(drop=True)
    tmp = df.dropna(subset=[feas]).reset_index(drop=True)

    n_clusters = cut_params.get('KMeans', 4)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300).fit(tmp)
    tmp['feassaxsxax'] = kmeans.fit_predict(tmp)
    st = tmp.groupby(['feassaxsxax']).agg({feas: [min, max]})
    bins = sorted(list(st[feas]['max']))
    bins[len(bins) - 1] = np.inf
    bins = [-np.inf] + bins

    return bins


def cut_feas_bytree(data, feas, target, cut_params={'max_depth': 4, 'thred': 2}):
    """
    cut feas by tree
    """
    df = data[[feas] + [target]].sort_values(feas).reset_index(drop=True)
    tmp = df.dropna(subset=[feas]).reset_index(drop=True)

    def tree_fit_bins(tmp, feas, target, cut_params):
        model = mf.tree_ana.TreeAna().fit_save_tree(tmp, target, feature_names=[feas],
                                                    filename=None,
                                                    max_depth=cut_params.get('max_depth', 3), \
                                                    min_samples_leaf=cut_params.get('min_samples_leaf', 0.05),
                                                    proportion=False)

        left = model.tree_.children_left
        idx = np.argwhere(left != -1)[:, 0]
        bins = list(model.tree_.threshold[idx])
        bins = sorted(bins + [tmp[feas].min(), tmp[feas].max()])

        return bins

    def merge_by_chi2(data, feas, target, bins):
        df, st, bins = cut_feas(data.dropna(subset=[feas]), feas, target, cut_params=bins)
        st['index'] = range(st.shape[0])
        st['good'] = st[feas] * (1 - st[target])
        st['bad'] = st[feas] * st[target]
        chi_list = []
        for i in range(len(st) - 1):
            chi_list.append(chi2_contingency(np.array(st[i:i + 2].iloc[:, -2:]))[0])
        st['chi'] = chi_list + [np.nan]
        min_chi = st['chi'].min()

        return st, min_chi, st.shape

    bins = tree_fit_bins(tmp, feas, target, cut_params)

    # get thred&max_bins
    thred = cut_params.get('thred', -1)
    max_bins = cut_params.get('max_bins', 9999)

    if len(bins) == 2:
        bins = tree_fit_bins(tmp, feas, target, {'max_depth': 1, 'min_samples_leaf': 0.01})
        thred, max_bins = -1, 9999

    st, min_chi, st_shape = merge_by_chi2(tmp, feas, target, bins)

    while (min_chi < thred) | (st_shape[0] > max_bins):
        try:
            idx = st[st['chi'] == min_chi]['index'].values[0]
            bins = bins[:idx + 1] + bins[idx + 2:]
        except IndexError:
            bins = bins
        st, min_chi, st_shape = merge_by_chi2(tmp, feas, target, bins)

    return bins


def cut_feas(data, feas, target, cut_params=20):
    """
    cut feas by cut,qcut,tree
    """
    ## get df ##
    if target is None:
        df = data[[feas]].sort_values(feas).reset_index(drop=True)
    else:
        df = data[[feas] + [target]].sort_values(feas).reset_index(drop=True)

    ## start cut ##
    if cut_params is None:
        # not cut
        bins = df[feas].unique().tolist()

    else:
        if isinstance(cut_params, list):
            bins = cut_params  # cut

        elif isinstance(cut_params, int):
            bins = pd.qcut(df[feas], cut_params, duplicates='drop', retbins=True)[1].tolist()  # qcut

        else:
            try:
                cut_params['KMeans']
                bins = cut_feas_bykmean(df, feas, cut_params)  # cut by kmean
            except:
                bins = cut_feas_bytree(df, feas, target, cut_params)  # cut by tree

        if len(bins) == 1:
            bins = [-np.inf, np.inf]
        else:
            bins[0], bins[len(bins) - 1] = -np.inf, np.inf
        df[feas] = pd.cut(df[feas], bins)

    # get st
    st1 = pd.DataFrame(df[feas].value_counts(dropna=False).sort_index())
    st1.index = st1.index.astype(str)
    df[feas] = df[feas].astype(str)

    if target is None:
        return df, st1, bins
    else:
        st2 = df.groupby([feas], dropna=False).agg({target: 'mean'})
        st = st1.merge(st2, left_index=True, right_index=True)
        return df, st, bins


class GetInformGain:
    '''
    get information_gain & information_gain_rto
    '''

    def get_entro(self, x):
        '''
        calc entro
        params x: is np.array
        '''
        entro = 0
        for x_value in set(x):
            p = np.sum(x == x_value) / len(x)
            if p != 0:
                logp = np.log2(p)
                entro -= p * logp
        return entro

    def get_condition_entro(self, x, y):
        '''
        calc condition_entro H(y|x)
        params x: is np.array
        params y: is np.array
        '''
        # calc condition_entro(y|x)
        condition_entro = 0
        for x_value in set(x):
            sub_y = y[x == x_value]
            temp_entro = self.get_entro(sub_y)
            condition_entro += (float(sub_y.shape[0]) / y.shape[0]) * temp_entro
        return condition_entro

    def get_information_gain(self, x, y):
        '''
        calc information_gain
        '''
        # information_gain
        inform_gain = (self.get_entro(y) - self.get_condition_entro(x, y))
        # information_gain_rto
        inform_gain_rto = inform_gain / self.get_entro(x)
        return [inform_gain, inform_gain_rto]


class WOE:
    '''
    calc woe & iv
    '''

    def __init__(self):
        self._WOE_MIN = -1.4
        self._WOE_MAX = 1.4

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def woe_single_x(self, x, y, event=1):
        event_total, non_event_total = self.count_binary(y, event=event)
        # x_labels = np.unique(x)
        x_labels = pd.Series(x).unique()
        woe_dict = {}
        iv = 0
        woe_ = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
            woe_ += abs(woe1)
        return eval(json.dumps(OrderedDict(sorted(woe_dict.items(), key=lambda x: x[1])))), iv, woe_


def rename_df(df, cols, prefix=None, suffix=None):
    '''
    rename cols
    '''
    if prefix is None:
        prefix = ''

    if suffix is None:
        suffix = ''

    cols_new = [prefix + c + suffix for c in cols]

    return df.rename(columns=dict(zip(cols, cols_new)))


def drop_cols_rf_corr(df_imp, cor_table, thre, metric_col):
    """
    drop cols ref corr and metric_col
    """
    drop_cols = []
    compare_cols = []
    corrs = []
    valu = cor_table[cor_table < 1].max().max()

    while valu > thre:
        tmp = pd.DataFrame(cor_table[cor_table == valu].max()).dropna()
        i = tmp.index.tolist()[0]
        j = tmp.index.tolist()[1]
        compare_cols.append('{}+{}'.format(i, j))
        corrs.append(valu)
        if df_imp[df_imp['feature'] == i][metric_col].tolist()[0] > df_imp[df_imp['feature'] == j][metric_col].tolist()[
            0]:
            drop_col = j
        else:
            drop_col = i

        # update cor_table
        drop_cols.append(drop_col)
        idx = (cor_table.index != drop_col)
        cor_table = cor_table[idx].drop(columns=drop_col)
        valu = cor_table[cor_table < 1].max().max()

    return pd.DataFrame({'compare_cols': compare_cols, 'corr': corrs, 'drop_cols': drop_cols})


def get_random_list(min_=0, max_=1, other_value=[], count=1, dtype=int):
    """
    构造random_list
    """

    # 构造list
    if dtype == int:
        x = list(np.random.randint(min_, max_, size=count)) + other_value
    # 构造float
    elif dtype == float:
        x = [np.random.uniform(min_, max_) * i for i in [1] * count] + other_value
    else:
        raise ValueError('only int/float')

    np.random.shuffle(x)

    return x


def get_pivot_table(data, index, columns, values, aggfunc):
    """
    get pivot table: 
    index is list
    columns is list 
    values is str or list
    aggfunc only support count,sum
    """
    # fillna
    if isinstance(values, str):
        df = data[index + columns + [values]]
    else:
        df = data[index + columns + values]
    for fea in df.columns:
        if re.search('category', str(df[fea].dtypes)):
            df[fea] = df[fea].cat.add_categories('na')
            df[fea].fillna('na', inplace=True)
        else:
            df[fea] = df[fea].fillna('na')

    tmp = pd.pivot_table(df, index=index,
                         columns=columns,
                         values=values, fill_value=0, dropna=False, aggfunc=aggfunc,
                         observed=False)

    return tmp


def get_pivot_table_af_proc(df, index, columns, values, aggfunc, is_color=True):
    """
    get pivot table now only support single index single columns and single values
    if is_color is False return dataframe else return style
    """

    def get_row_column_sum(st):
        st['Row_sum'] = st.sum(axis=1)
        st.loc['Column_sum'] = st.sum(axis=0)
        st = st.fillna(st['Row_sum'][:-1].sum())
        return st

    dict_ = {}
    for func in ['count', 'sum']:
        st = get_pivot_table(df, index, columns, values, func)
        st = get_row_column_sum(st)
        dict_.update({func: st})
    dict_.update({'mean': (dict_['sum'] / dict_['count']).round(4)})

    for key, values_ in dict_.items():
        values_.columns = [[key] * len(values_.columns), values_.columns]
        dict_.update({key: values_})

    # get mean columns
    mean_columns = dict_['mean'].columns

    # out
    if isinstance(aggfunc, str):
        out = dict_[aggfunc]
    else:
        isinstance(aggfunc, list)
        for i, func in enumerate(aggfunc):
            if i == 0:
                out = dict_[func]
            else:
                out = out.merge(dict_[func], left_index=True, right_index=True)

    # color 
    if is_color is True:
        try:
            out[mean_columns]
            out = out.style.background_gradient(cmap='RdYlGn', low=0.7, high=0, vmin=0,
                                                subset=mean_columns)
        except:
            pass

    return out


def get_confusion_matrix(df, score, target, thred):
    tn, fp, fn, tp = confusion_matrix(df[target].values, [0 if x < thred else 1 for x in df[score].values]).ravel()

    display.display(
        pd.DataFrame(np.array([[f'{tp}(tp)', f'{fn}(fn)'], [f'{fp}(fp)', f'{tn}(tn)']]), index=[["True", "True"],
                                                                                                ["POSITIVE",
                                                                                                 "NEGTIVE"]],
                     columns=[["PREDIT", "PREDIT"], ["POSITIVE", "NEGTIVE"]]))

    print(f'查准率={round(tp / (tp + fp) * 100, 2)}%')
    print(f'召回率={round(tp / (tp + fn) * 100, 2)}%')


def time_conversion(data, input_col, output_col, input_format, output_format):
    """
    output_format like "%Y-%m-%d %H:%M:%S" , int
    input_format like "%Y-%m-%d %H:%M:%S" , int
    """
    if input_format is int:
        data[output_col] = data[input_col].apply(lambda x: time.strftime(output_format, time.localtime(x)))

    else:
        data[input_col] = data[input_col].astype(str)
        if output_format is int:
            data[output_col] = data[input_col].apply(lambda x: int(time.mktime(time.strptime(x, input_format))))
        else:
            data[output_col] = data[input_col].apply(lambda x: pd.to_datetime(x).strftime(output_format))

    return data


def get_day_range_step(start, end, day, freq):
    """
    params start, end: like '20200504','2020-05-04'
    params freq: like d,3d,M,2M,Y
    """
    if len([v for v in [start, end, day] if v is not None]) < 2:
        print('we need at least two parameters')

    else:
        if (start is None) or (pd.isnull(start)):
            start = pd.to_datetime(end) - dt.timedelta(days=day)
        elif (end is None) or (pd.isnull(end)):
            end = pd.to_datetime(start) + dt.timedelta(days=day)

        return [str(i.date()) for i in pd.date_range(start=start, end=end, freq=freq).tolist()]


def getfile_size_(file):
    plain_size = float(os.path.getsize(join(file)))
    if plain_size <= 1024:
        return str(round(plain_size, 2)) + 'B'
    if plain_size <= 1024 * 1024:
        return str(round(plain_size / 1024, 2)) + 'K'
    if plain_size <= 1024 * 1024 * 1024:
        return str(round(plain_size / 1024 / 1024, 2)) + 'M'
    if plain_size <= 1024 * 1024 * 1024 * 1024:
        return str(round(plain_size / 1024 / 1024 / 1024, 2)) + 'G'


def getfile_time_(file):
    time_ = int(os.path.getmtime(file))
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_))


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return None

    else:
        cfg = {}
        for filename in dir_list:
            cfg.update({filename: {'desc': {'save_time': getfile_time_(join(file_path, filename)),
                                            'size': getfile_size_(join(file_path, filename))}}})

        data = pd.DataFrame.from_dict(cfg, orient='index', dtype=None, columns=None).reset_index().rename(
            columns={'index': 'filename'})
        return data.merge(pd.json_normalize(data['desc'].map(lambda x: x), sep='_'),
                          left_index=True, right_index=True).drop(columns=['desc'], axis=1)


def make_dir(filepath):
    try:
        os.stat(filepath)
    except FileNotFoundError:
        os.mkdir(filepath)


def groupby_and_rename(data, by, agg_func, rename=None, is_flat=False):
    """
    groupby and rename multicolumns
    for example:
        if agg_func = {'money':[len,sum,'mean'],'feas':max}
           rename = {'金额':['数量','总和','均值'],'feas':'最大值'}
           or rename = [['金额','金额','金额','feas'],['数量','总和','均值','最大值']] 
        
        if agg_func = {'feas':max}
           rename = {'feas':'特征'}
    """
    st = data.groupby(by, dropna=False).agg(agg_func)

    if rename is not None:
        # rename MultiIndex
        if re.search('MultiIndex', str(st.columns)):
            # when rename is dict
            if isinstance(rename, dict):
                keys_list = []
                values_list = []
                for keys, value in rename.items():
                    if isinstance(value, list):
                        keys_list.extend([keys] * len(value))
                        values_list.extend(value)
                    else:
                        keys_list.extend([keys])
                        values_list.append(value)
                rename = [keys_list, values_list]

                # rename MultiIndex columns
            st.columns = pd.MultiIndex.from_arrays(rename)

        # rename columns
        else:
            st.rename(columns=rename, inplace=True)

    # flat columns
    if (re.search('MultiIndex', str(st.columns)) is not None) & (is_flat is True):
        st.columns = [f'{st.columns[i][0]}_{st.columns[i][1]}' for i in range(len(st.columns))]

    return st


class GetOvdInform:
    """
    get ovd information
    """

    def _calc_ovd_days(self, df, pay_date, finish_date, ref_date):
        """
        repay = _calc_ovd_days(repay, pay_date='PAYDATE', finish_date='FINISHDATE', 
                    ref_date=pd.to_datetime((dt.datetime.now()).strftime("%Y-%m-%d"))- dt.timedelta(days=0))
        """
        # to_datetime
        for fea in [pay_date, finish_date]:
            df[fea] = pd.to_datetime(df[fea])
        df[f'{finish_date}_af_proc'] = df[finish_date]
        idx = (df[pay_date] < ref_date) & (pd.isnull(df[finish_date]))
        df.loc[idx, f'{finish_date}_af_proc'] = ref_date
        df['ovd_days'] = (df[f'{finish_date}_af_proc'] - df[pay_date]).dt.days
        return df

    def _get_ovd_inform(self, ovd_days_list, money_list, pay_date_list,
                        finish_date_list, thred):
        """
        get max ovd cnt & pessimistic ovd money
        
        input:
        params ovd_days_list: like [0,3,5]，其中0表示未逾期，3表示逾期3天
        params money_list: like [100,120,100]
        params thred: define bad users like 3，表示3+
        
        return
        ovd_cnt_list: 前n期最大逾期天数是否为thre，如[0,1...] 前1期最大逾期天数不为thred，前2期最大逾期天数为thred
        ovd_money_list: 计算悲观逾期金额，即当期逾期，为默认后面的金额都逾期 
        """

        i, ovd_period, ovd_max, ovd_period_list, ovd_max_list = 0, 0, 0, [], []
        ovd_money_pessimistic_list, ovd_money_optimistic_list = [], []
        while i < len(ovd_days_list):
            # calc ovd_period & ovd_max
            if ovd_days_list[i] <= thred:
                ovd_period = 0
                ovd_max = max(ovd_max, 0)
            elif ovd_days_list[i] > thred:
                ovd_period, ovd_max = 1, 1
            else:
                ovd_period, ovd_max = np.nan, np.nan

            ovd_period_list.append(ovd_period)
            ovd_max_list.append(ovd_max)

            # calc ovd money
            # 过去未还
            ovd_money_optimistic = sum(np.array(money_list)[np.where(
                np.array(finish_date_list[:i + 1]) > pay_date_list[i] + dt.timedelta(days=thred))]) \
                if pd.notnull(ovd_days_list[i]) else np.nan
            # 未来未还
            norepay_feature = sum(money_list[i + 1:])
            ovd_money_pessimistic = (ovd_days_list[i] > thred) * (ovd_money_optimistic + norepay_feature) if pd.notnull(
                ovd_days_list[i]) else np.nan

            ovd_money_pessimistic_list.append(ovd_money_pessimistic)
            ovd_money_optimistic_list.append(ovd_money_optimistic)

            i = i + 1

        return ovd_period_list, ovd_max_list, ovd_money_pessimistic_list, ovd_money_optimistic_list

    def get_ovd_inform(self, df, by, period_col, pay_date_col, finish_date_col,
                       money_col, ovd_col, thred_list):
        """
        repay = get_ovd_inform(repay, by='OBJECTNO', period_col='PERIODNO',
                               pay_date_col='PAYDATE', finish_date_col='FINISHDATE_af_proc',
                               money_col='money', ovd_col='ovd_days', thred_list=[1,3,5,10])
        """

        # get ovd_list & money_list
        df = df.sort_values([by, period_col]).reset_index(drop=True)
        tmp = df.groupby(by).apply(lambda x: tuple([list(x[ovd_col]), list(x[money_col]),
                                                    list(x[pay_date_col]), list(x[finish_date_col])])) \
            .to_frame('tmp_col').reset_index()
        tmp = tmp.sort_values([by]).reset_index(drop=True)

        for thred in tqdm(thred_list):
            result = tmp['tmp_col'].apply(lambda x: self._get_ovd_inform(ovd_days_list=x[0], money_list=x[1],
                                                                         pay_date_list=x[2], finish_date_list=x[3],
                                                                         thred=thred))

            ovd_period_list, ovd_max_list, ovd_money_pessimistic_list, ovd_money_optimistic_list = [], [], [], []
            for value in result:
                ovd_period_list.extend(value[0])
                ovd_max_list.extend(value[1])
                ovd_money_pessimistic_list.extend(value[2])
                ovd_money_optimistic_list.extend(value[3])

            df[f'ovd{thred}'] = ovd_period_list
            df[f'ovd{thred}_max'] = ovd_max_list
            df[f'ovd{thred}_optimistic_money'] = ovd_money_optimistic_list
            df[f'ovd{thred}_pessimistic_money'] = ovd_money_pessimistic_list

        return df


class ExcelWrite:
    """
    insert value/df/pic in excel
    """

    def __init__(self):
        """
        style_dict 说明
        # 位置说明
            # VERT_TOP = 0x00 上端对齐
            # VERT_CENTER = 0x01 居中对齐（垂直方向上）
            # VERT_BOTTOM = 0x02 低端对齐
            # HORZ_LEFT = 0x01 左端对齐
            # HORZ_CENTER = 0x02 居中对齐（水平方向上）
            # HORZ_RIGHT = 0x03 右端对齐

        # 颜色说明
            # https://blog.csdn.net/guoxinian/article/details/80242353
            如：'lightskyblue':         '#87CEFA',
                'lemonchiffon':         '#FFFACD',
                'lightgray':            '#D3D3D3',
                'lightpink':            '#FFB6C1',
                'bisque':               '#FFE4C4',
        """
        # english_width，chinese_width
        self.english_width = 0.12
        self.chinese_width = 0.21

        # get Alphabet_list
        letter1, letter2 = [chr(ord('A') + i) for i in range(26)], []
        for i in letter1:
            for j in letter1:
                letter2.append(i + j)
        self.Alphabet_list = letter1 + letter2

        # get style
        self.orig_style = {'border': 1,  # 边框
                           'align': 'left',  # 左对齐
                           'valign': 'vcenter',  # 垂直居中
                           'bold': False,  # 加粗（默认False）
                           'underline': False,  # 下划线
                           'italic': False,  # 斜体字
                           'font': u'宋体',  # 字体 'Times New Roman'
                           # 'fg_color': '#00868B', # 背景色
                           'color': 'black',  # 字体颜色
                           'size': 10}

    def get_workbook_sheet(self, workbook, work_sheet_name):
        """
        excel中新增sheet
        """
        worksheet = workbook.add_worksheet(work_sheet_name)
        return workbook, worksheet

    def check_contain_chinese(self, check_str):
        """
        判断字符串中是否含有中文，True表示中文，False表示非中文
        """
        out = []
        for ch in str(check_str).encode('utf-8').decode('utf-8'):
            if u'\u4e00' <= ch <= u'\u9fff':
                out.append(True)
            else:
                out.append(False)
        return out, len(out) - sum(out), sum(out)

    def insert_value2table(self, workbook, worksheet, value, insert_space, style={},
                           decimal_point=4, is_set_col=True):
        """
        在表中特定位置填入值
        """
        # update style
        style_dict = self.orig_style
        style_dict.update(style)
        style = workbook.add_format(style_dict)

        # insert value
        try:
            value = round(float(value), decimal_point)
        except:
            pass

        if pd.isnull(value):
            value = 'nan'
        if re.search('tuple|list|numpy.dtype', str(type(value))):
            value = str(value)

        _ = worksheet.write(insert_space, value, style)

        # get start_col
        start_col = insert_space[:re.search('(\d+)', insert_space).span()[0]]

        # set_column(start_colum, end_column)
        try:
            orig_col_size = worksheet.col_sizes[self.Alphabet_list.index(start_col)][0]
        except KeyError:
            orig_col_size = 0

        if is_set_col is True:
            _ = worksheet.set_column(self.Alphabet_list.index(start_col), self.Alphabet_list.index(start_col),
                                     max([(self.check_contain_chinese(value)[1] * self.english_width +
                                           self.check_contain_chinese(value)[2] * self.chinese_width) * style_dict[
                                              'size'],
                                          10, orig_col_size]))
        else:
            _ = worksheet.set_column(self.Alphabet_list.index(start_col), self.Alphabet_list.index(start_col),
                                     max([10, orig_col_size]))

        return workbook, worksheet

    def calc_continuous_cnt(self, list_, index_=0):
        if index_ >= len(list_):
            return None, None, None

        else:
            cnt, str_ = 0, list_[index_]
            for i in range(index_, len(list_), 1):
                if list_[i] == str_:
                    cnt = cnt + 1
                else:
                    break
            return str_, index_, cnt

    def merge_df(self, workbook, worksheet, df, insert_space, style):

        # get start_row,start_col
        start_row = int(insert_space[re.search('(\d+)', insert_space).span()[0]:])
        start_col = insert_space[:re.search('(\d+)', insert_space).span()[0]]
        add_col = len(df.index.names)
        add_row = len(df.columns.names)

        # merge columns
        if re.search('MultiIndex', str(type(df.columns))):
            for site in range(len(df.columns.names)):
                list_ = [c[site] for c in df.columns]
                str_, index_, count_ = self.calc_continuous_cnt(list_)
                while str_ is not None:
                    if count_ == 1:
                        workbook, worksheet = self.insert_value2table(workbook, worksheet, value=str_,
                                                                      insert_space=f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + add_col + index_]}{start_row + site}',
                                                                      style=style)
                    else:
                        worksheet.merge_range(
                            f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + add_col + index_]}{start_row + site}' + ':'
                                                                                                                                f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + add_col + index_ + count_ - 1]}{start_row + site}',
                            str_, cell_format=workbook.add_format(style))
                    str_, index_, count_ = self.calc_continuous_cnt(list_, index_ + count_)

                    # merge index
        if re.search('MultiIndex', str(type(df.index))):
            for site in range(len(df.index.names)):
                list_ = [c[site] for c in df.index]
                str_, index_, count_ = self.calc_continuous_cnt(list_)
                while str_ is not None:
                    if count_ == 1:
                        workbook, worksheet = self.insert_value2table(workbook, worksheet, value=str_,
                                                                      insert_space=f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + site]}{start_row + add_row + index_}',
                                                                      style=style)
                    else:
                        worksheet.merge_range(
                            f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + site]}{start_row + add_row + index_}' + ':'
                                                                                                                                f'{self.Alphabet_list[self.Alphabet_list.index(start_col) + site]}{start_row + add_row + index_ + count_ - 1}',
                            str_, cell_format=workbook.add_format(style))
                    str_, index_, count_ = self.calc_continuous_cnt(list_, index_ + count_)

        return workbook, worksheet

    def insert_df2table(self, workbook, worksheet, df, insert_space, style={},
                        is_merge=True):
        """
        表格插入到excel中
        """
        # style
        style_dict = self.orig_style
        style_dict.update(style)

        # get start_row,start_col
        start_row = int(insert_space[re.search('(\d+)', insert_space).span()[0]:])
        start_col = insert_space[:re.search('(\d+)', insert_space).span()[0]]

        # proc df
        tmp = df.reset_index().drop(columns='index', errors='ignore')
        insert_df = pd.concat([pd.DataFrame(np.array(list(tmp))).T, pd.DataFrame(np.array(tmp))], axis=0).reset_index(
            drop=True)

        # insert df
        for i in range(insert_df.shape[0]):
            for j in range(insert_df.shape[1]):
                value = insert_df.loc[i][j]
                workbook, worksheet = self.insert_value2table(workbook, worksheet, value=value,
                                                              insert_space=self.Alphabet_list[self.Alphabet_list.index(
                                                                  start_col) + j] + \
                                                                           str(start_row + i), style=style_dict)
        # merge
        if is_merge is True:
            style_dict.update({'align': 'center'})
            workbook, worksheet = self.merge_df(workbook, worksheet, df, insert_space, style_dict)

        return workbook, worksheet

    def insert_pic2table(self, workbook, worksheet, pic_name, insert_space,
                         style={'x_scale': 0.5, 'y_scale': 0.5, 'x_offset': 0.2, 'y_offset': 0.2}):
        """
        图片插入到excel中
        """
        _ = worksheet.insert_image(insert_space, pic_name, style)
        return workbook, worksheet


class ENCRYPT:

    @classmethod
    def fun_md5(cls, dt):
        m = hashlib.md5()
        m.update(str(dt).strip().encode('utf-8'))
        return m.hexdigest()

    @classmethod
    def get_md5(cls, out, cols=['name', 'id_number', 'mobile_number']):
        user_md5 = out[cols].applymap(lambda dt: fun_md5(dt) if dt != '' else '')
        user_md5.columns = ['{}_md5'.format(c) for c in cols]
        out = out.merge(user_md5, left_index=True, right_index=True)
        return out

    @classmethod
    def jm_sha256_single(cls, value):
        """
        sha256加密
        :param value: 加密字符串
        :return: 加密结果转换为16进制字符串，并大写
        """
        hsobj = hashlib.sha256()
        hsobj.update(value.encode("utf-8"))
        return hsobj.hexdigest().upper()

    @classmethod
    def get_sha256(cls, df, cols=['name', 'id_number', 'mobile_number']):
        for f in cols:
            df[f] = df[f].astype(str)
            df['{}_sha256'.format(f)] = df[f].apply(lambda dt: jm_sha256_single(dt) if dt != '' else '')
        return df

    @classmethod
    def sm3_dt(cls, dt):
        data = str(len(dt)) + dt
        result = sm3.sm3_hash(func.bytes_to_list(data.encode('utf-8')))
        return result

    @classmethod
    def get_sm3(cls, df, cols):
        for f in cols:
            df[f] = df[f].astype(str)
            df['{}_sm3'.format(f)] = df[f].apply(lambda x: sm3_dt(x))

        return df
