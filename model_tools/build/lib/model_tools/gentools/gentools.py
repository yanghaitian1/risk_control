import datetime as dt
import os
import random
import re
import string
import time
from itertools import islice
from os.path import join
from itertools import tee
import numpy as np
import pandas as pd
from IPython import display
from pip._internal import pep425tags
from sklearn.metrics import confusion_matrix


def show_supproted_whl():
    return display.display(pep425tags.get_supported())


def extend_(value):
    """
    Extend elements in value

    :param value: list
    :return:

    Examples:
        extend_(value = [[1,2,3],'asx',[23]])
    """

    out = []
    for value_ in value:
        if isinstance(value_, list):
            out.extend(value_)
        else:
            out.append(value_)
    return out


def append_(value, lice):
    """
    append elements in value according to lice

    :param value: list
    :param lice: list
    :return:

    Examples:
        append_(value=[1,2,3,'asx','ca',2], lice=[3,1,2])
    """

    it = iter(value)
    return [list(islice(it, size)) for size in lice]


def pairwise(iterable):
    """
    convert list to set

    :param iterable:  list
    :return:

    Examples:
        array = np.array([['ely_gt3_repay_cnt_rf_repay_p66d','ely_gt3_repay_cnt_rf_repay_p96d','ely_gt3_repay_cnt_rf_repay_p186d'],
                        ['normal_repay_amt_rf_repay_sum_p36d','normal_repay_amt_rf_repay_sum_p66d','normal_repay_amt_rf_repay_sum_p96d']])
        pat = 'p[0-9]*(d)'

        for iterable in array:
            for c0, c1 in pairwise(iterable):
                col_new = '{}{}/{}_rto'.format(re.sub(pat, '', c0), re.search(pat, c1).group(), re.search(pat, c0).group())
                x_train[col_new] = x_train[c1]/x_train[c0]
                # print(c1,c0,col_new)
    """

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def eval_value(value):
    """
    Eval value

    :param value: int/float/list/dict/str
    :return: eval value

    Examples:
        (1) if input value is str:'x_train', then return dataframe x_train
        (2) if input value is str:'max', then return function max
    """

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


def rename_df(df, cols, prefix=None, suffix=None):
    """
    Rename cols according to the specified prefix and suffix

    :param df: dataframe
    :param cols: original feature name
    :param prefix: add according prefix
    :param suffix: add according suffix
    :return: df to with modified name according to the specified prefix and suffix
    """

    if prefix is None:
        prefix = ''

    if suffix is None:
        suffix = ''

    cols_new = [prefix + c + suffix for c in cols]

    return df.rename(columns=dict(zip(cols, cols_new)))


def get_random_object(min_=0, max_=1, other_value=[], count=1, dtype=int):
    """
    Get random object

    :param min_: use when dtype is int or float min_thred
    :param max_: use when dtype is int or float max_thred
    :param other_value: specifical value like [-9999,np.nan]
    :param count: len of list or str
    :param dtype: support int/float/str
    :return: return str or list

    Examples:
        (1) get_random_object(10,100,[-2,np.nan,np.nan],4,int)
            return [66, 29, 41, nan, -2, nan, 99]
        (2) get_random_object(10,100,[-2,np.nan,np.nan],4,float)
            return [nan,52.089146468370956,92.49709466309213,59.53310530471421,nan,-2,75.493189558075]
        (3) get_random_object(10,100,[-2,np.nan,np.nan],4,str)
            return rxbF
    """

    if dtype == str:  # only params: count is useful
        # return ''.join(random.sample(string.ascii_letters + string.digits, count))
        return ''.join(random.sample(string.ascii_letters, count))

    else:
        if other_value is None:
            other_value = []
        if dtype == int:
            x = list(np.random.randint(min_, max_, size=count)) + other_value
        elif dtype == float:
            x = [np.random.uniform(min_, max_) * i for i in [1] * count] + other_value
        else:
            raise ValueError('only int/float')

        np.random.shuffle(x)
        return x


def df_fillna(data, fea, fill_value='NaN', inplace=False):
    """
    Always use when df[fea].dtypes is CategoricalDtype

    :param df: dataframe
    :param fea: str, fea_name
    :param fill_value: fill_value
    :param inplace: support True/False
    :return: if inplace is True return None elif inplace is False return Series

    Examples:
        df = pd.DataFrame({'feas1':['A','A','B',np.nan,'C','B',np.nan,'A','B','C'],
                      'feas2':[1,0,5,np.nan,6,7,2,3,2,np.nan]})
        df['feas3'] = pd.cut(df['feas2'], [-np.inf, 2, 5, np.inf])
        df_fillna(df, 'feas3', fill_value='NaN',inplace=True)
    """

    df = data[[fea]]

    if re.search('category', str(df[fea].dtypes)):
        try:
            df[fea] = df[fea].cat.add_categories(fill_value)
        except ValueError:
            # new categories must not include old categories: means has already add_categories(fill_value)
            pass
    df[fea] = df[fea].fillna(fill_value)

    if inplace:
        data[fea] = df[fea]
        return None
    else:
        return df[fea]


def keep_df_value(df, feas, drop_value=None):
    """
    Drop feas value prepare for calculation of various indicators or modeling

    :param df: dataframe
    :param feas: feaa in dataframe
    :param drop_value: int/float/list/callable value to drop
    :return: dataframe after drop value

    Examples:
        (1) mask,tmp = keep_df_value(df, feas, drop_value=0)
        (2) mask,tmp = keep_df_value(df, feas, drop_value=[0,1,11])
        (3) mask,tmp = keep_df_value(df, feas, drop_value=lambda x: x>=10)
        (4) mask,tmp = keep_df_value(df, feas, drop_value=None)
        (5) mask,tmp = keep_df_value(df, feas, drop_value=np.nan)
    """

    if drop_value is np.nan:
        mask = pd.isnull(df[feas])
    elif isinstance(drop_value, int) or isinstance(drop_value, float):
        mask = df[feas] == drop_value
    elif isinstance(drop_value, list):
        mask = df[feas].isin(drop_value)
    elif callable(drop_value):
        mask = df[feas].apply(drop_value)
    else:
        mask = pd.Series([False] * len(df))

    return ~mask, df[~mask].reset_index(drop=True)


def SelectKBest(values, SelectKBest, thred=-1, ascending=False, key=None):
    """
    Select Best feaures according to feature values

    :param values: pd.Series
    :param SelectKBest: float if 0<SelectKBest<=1 means select SelectKBest% feastures, if SelectKBest>1 means select SelectKBest features
    :param thred: select features which values>thred
    :param ascending: sort_values(ascending=ascending)
    :param key: sort_values(key=key)
    :return: index which meet the conditions

    Examples:
        df = pd.DataFrame({'feas':['feas1','feas2','feas3','feas4','feas5'
                          ,'feas6','feas7','feas8','feas9','feas10'],
                    'value':[0.1,0.2,0.15,-0.3,0.4
                            ,0.22,-0.19,0.64,0.22,0.11]}).set_index('feas')
        (1) SelectKBest(values=df['value'], SelectKBest=0.5, thred=0.03, ascending=False, key=abs)
        (2) SelectKBest(values=df['value'], SelectKBest=3, thred=0.5, ascending=False, key=None)
        (3) SelectKBest(values=df['value'], SelectKBest=8, thred=-1, ascending=True, key=None)
    """

    if SelectKBest <= 1:
        SelectKBest = int(np.ceil(len(values) * SelectKBest))

    # SelectKBest
    values = values.sort_values(ascending=ascending, key=key)[:SelectKBest]
    # Select by thred
    if key == abs:
        return list(values[abs(values) > thred].index)
    else:
        return list(values[values > thred].index)


def groupby_and_rename(data, by, agg_func, rename=None, is_flat=False):
    """
    Groupby according to aggfunc and rename columns based on rename

    :param data: dataframe
    :param by: groupby feas
    :param agg_func: agg_func in groupby like {'money':[len,sum,'mean'],'feas':max}/{'feas':max}
    :param rename: list or dict like {'金额':['数量','总和','均值'],'feas':'最大值'}
                                     [['金额','金额','金额','feas'],['数量','总和','均值','最大值']]
    :param is_flat: True/False
    :return: groupby dataframe witch meets the requirements

    Examples:
        (1) by = 'month'
            groupby_and_rename(x_train, by, agg_func={'p1_0':[len,sum,'mean'],'money':sum},
                         rename={'p1_0':['到期数','命中数','占比'],'放款金额':'总和'}, is_flat=True)
        (2) x_train['ovd_days_rf_repay_max_p186d_proc'] = pd.cut(x_train['ovd_days_rf_repay_max_p186d'],[-np.inf,1,2,np.inf])
            by = ['month','ovd_days_rf_repay_max_p186d_proc']
            groupby_and_rename(x_train, by, agg_func={'p1_0':[len,sum,'mean'],'money':sum},
                                     rename={'p1_0':['到期数','命中数','占比'],'放款金额':'总和'}, is_flat=True)
    """

    if isinstance(by, str):
        by = [by]
    for fea in by:
        if re.search('category', str(data[fea].dtypes)):
            if sum(pd.isnull(data[fea])) > 0:
                df_fillna(data, fea, fill_value='NaN', inplace=True)

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


def get_pivot_table(data, index, columns, values, aggfunc, dropna=False):
    """
    Dropna in pd.pivot_table can not work, get pivot table to support dropna

    :param data: dataframe
    :param index: str/list: index in pivot table
    :param columns: str/list: columns in pivot table
    :param values:  str/list: values in pivot table
    :param aggfunc: aggfunc support ['count','sum','mean']
    :param dropna: True/False
    :return: pivot table

    Examples:
        (1) index = 'ovd_days_rf_due_sum_p66d'
            columns = 'ovd_days_rf_due_sum_p96d'
            target = ['p1_0','p1_1']
            st = get_pivot_table(x_train, index=index, columns=columns,
                            values=target, aggfunc=['count','sum','mean'], dropna=False)

        (2) x_train['ovd_days_rf_due_sum_p66d_proc'] = pd.cut(x_train['ovd_days_rf_due_sum_p66d'], [-np.inf,0,1,np.inf])
            x_train['ovd_days_rf_due_sum_p96d_proc'] = pd.cut(x_train['ovd_days_rf_due_sum_p96d'], [-np.inf,0,1,np.inf]
            index = ['ovd_days_rf_due_sum_p66d_proc','ord_cnt_max_in1d_is_holiday_p96d']
            columns = 'ovd_days_rf_due_sum_p96d_proc'
            target = ['p1_0']
            st = get_pivot_table(x_train, index=index, columns=columns,
                            values=target, aggfunc=['sum','mean'], dropna=False)
    """

    if isinstance(index, str):
        index = [index]
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(values, str):
        values = [values]

    df = data[index + columns + values]
    if dropna is False:
        for fea in df.columns:
            if sum(pd.isnull(df[fea])) > 0:
                df_fillna(df, fea, fill_value='NaN', inplace=True)

    st = pd.pivot_table(df, index=index, columns=columns,
                        values=values, fill_value=0, dropna=dropna,
                        aggfunc=aggfunc, observed=False)

    return st


def get_pivot_table_af_proc(df, index, columns, values, aggfunc, is_color=False):
    """
    get pivot table with row_number and column_sum but only support one index, column and target

    :param df: dataframe
    :param index: str/list: index in pivot table
    :param columns: str/list: columns in pivot table
    :param values: str/list: values in pivot table
    :param aggfunc: aggfunc support ['count','sum','mean']
    :param is_color: is_color support True/False
    :return: pivot table

    Examples:
        x_train['ovd_days_rf_due_sum_p66d_proc'] = pd.cut(x_train['ovd_days_rf_due_sum_p66d'], [-np.inf,0,1,np.inf])
        x_train['ovd_days_rf_due_sum_p96d_proc'] = pd.cut(x_train['ovd_days_rf_due_sum_p96d'], [-np.inf,0,1,np.inf])
        (1) index = ['ovd_days_rf_due_sum_p66d_proc']
            columns = 'ovd_days_rf_due_sum_p96d_proc'
            target = ['p1_0']
            st = get_pivot_table_af_proc(x_train, index=index, columns=columns,
                            values=target, aggfunc=['count','sum','mean'],is_color=False)

        (2) index = ['ord_cnt_max_in1d_is_holiday_p96d']
            columns = ['ovd_days_rf_due_sum_p96d_proc']
            target = 'p1_1'
            st = get_pivot_table_af_proc(x_train, index=index, columns=columns,
                            values=target, aggfunc=['sum','mean'],is_color=True)
    """

    def get_row_column_sum(st):
        try:
            st['Row_sum'] = st.sum(axis=1)
        except TypeError:
            st.columns = pd.MultiIndex.from_arrays([[c[0] for c in st.columns], [str(c[1]) for c in st.columns]])
            st['Row_sum'] = st.sum(axis=1)

        try:
            st.loc['Column_sum'] = st.sum(axis=0)
        except TypeError:
            st.index = [str(c) for c in st.index.categories]
            st.loc['Column_sum'] = st.sum(axis=0)

        st = st.fillna(st['Row_sum'][:-1].sum())
        return st

    st = get_pivot_table(df, index, columns, values=values,
                         aggfunc=['count', 'sum', 'mean'], dropna=False)

    # add row_sum & column_sum
    out = pd.concat([get_row_column_sum(st['count']), get_row_column_sum(st['sum'])], axis=1)
    cnt = len(st.columns) // 3 + 1
    out = pd.concat([out, out.iloc[:, cnt:2 * cnt] / out.iloc[:, :cnt]], axis=1)
    rename = [['count'] * cnt + ['sum'] * cnt + ['mean'] * cnt, [c[0] for c in out.columns],
              [c[1] for c in out.columns]]
    out.columns = pd.MultiIndex.from_arrays(rename)
    out = out[aggfunc]

    # color 
    if is_color is True:
        try:
            out['mean']
            out = out.style.background_gradient(cmap='RdYlGn', low=0.7, high=0, vmin=0, subset='mean')
        except:
            pass

    return out


def get_confusion_matrix(df, score, target, thred):
    """
    Calc tn,fp,fn,tp and recall,precision

    :param df: dataframe
    :param score: prediction
    :param target: truth
    :param thred: prediction thred
    :return: confusion_matrix

    Examples:
        get_confusion_matrix(x_train, score='ovd_days_rf_due_sum_p66d', target='p1_0', thred=1)
    """

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
    Convert time according to input and output format

    :param data: dataframe
    :param input_col: input_col name
    :param output_col: output_col name
    :param input_format: input format like int, "%Y-%m-%d %H:%M:%S"
    :param output_format: output format like int, "%Y-%m-%d %H:%M:%S"
    :return: time after conversion

    Examples:
        data = pd.DataFrame({'date1':['2021-01-01 00:00:05','2021-03-01 02:00:05','2022-01-01 10:00:05',np.nan],
                             'date2':[1630660999, np.nan,1630655999, 1630560999],
                             'date3':['2021/01/01','2021/04/05',np.nan, '2022/01/01'],
                             'date4':['20210101','20210405',np.nan, '20221211'],
                             'date5':[np.nan, pd.to_datetime('2021-01-01 00:00:05'), pd.to_datetime('2021-03-02 10:00:05'),
                                     pd.to_datetime('2021-12-01 00:20:05')]})

        (1) time_conversion(data, input_col='date1', output_col='date_proc', input_format="%Y-%m-%d %H:%M:%S", output_format="%Y%m%d")
        (2) time_conversion(data, input_col='date2', output_col='date_proc', input_format=int, output_format="%Y/%m/%d %H:%M:%S")
        (3) time_conversion(data, input_col='date3', output_col='date_proc', input_format="%Y/%m/%d", output_format="%Y-%m-%d %H:%M:%S")
        (4) time_conversion(data, input_col='date4', output_col='date_proc', input_format="%Y%m%d", output_format="%Y-%m-%d %H:%M:%S")
        (5) time_conversion(data, input_col='date5', output_col='date_proc', input_format="%Y-%m-%d %H:%M:%S", output_format="%Y/%m/%d")
    """

    if input_format is int:
        data[output_col] = data[input_col].apply(
            lambda x: time.strftime(output_format, time.localtime(x)) if pd.notnull(x)
            else np.nan)

    else:
        if output_format is int:
            data[output_col] = data[input_col].apply(
                lambda x: int(time.mktime(time.strptime(str(x), input_format))) if pd.notnull(x)
                else np.nan)
        else:
            data[output_col] = data[input_col].apply(
                lambda x: pd.to_datetime(str(x)).strftime(output_format) if pd.notnull(x)
                else np.nan)

    return data


def get_day_range_step(start, end, day, freq):
    """
    Given start time, end time, time interval and time step to get date range

    :param start: start time like '20200504','2020-05-04'
    :param end: end time like '20200504','2020-05-04'
    :param day: time interval like 10,30
    :param freq: time step like d,3d,M,2M,Y
    :return: time range according to input

    Examples:
        (1) get_day_range_step(start='2021-08-20', end='20210825', day=None, freq='d')
        (2) get_day_range_step(start='20210720', end='2021/09/25', day=None, freq='m')
        (3) get_day_range_step(start='20000120', end='2021/09/25', day=None, freq='10Y')
    """

    if len([v for v in [start, end, day] if v is not None]) < 2:
        print('we need at least two parameters')

    else:
        if (start is None) or (pd.isnull(start)):
            start = pd.to_datetime(end) - dt.timedelta(days=day)
        elif (end is None) or (pd.isnull(end)):
            end = pd.to_datetime(start) + dt.timedelta(days=day)

        return [str(i.date()) for i in pd.date_range(start=start, end=end, freq=freq).tolist()]


def make_dir(filepath):
    """
    Make dir if filepath does not exist

    :param filepath: folder name
    :return: create new folder named filepath
    """

    try:
        os.stat(filepath)
    except FileNotFoundError:
        os.mkdir(filepath)


def _getfile_size(file):
    """
    Get file size

    :param file: file_path+file_name
    :return: file size

    Examples:
        _getfile_size('E:/atome/git/data-tools/data_tools/third_features/create_table.py')
    """

    plain_size = float(os.path.getsize(join(file)))
    if plain_size <= 1024:
        return str(round(plain_size, 2)) + 'B'
    if plain_size <= 1024 * 1024:
        return str(round(plain_size / 1024, 2)) + 'K'
    if plain_size <= 1024 * 1024 * 1024:
        return str(round(plain_size / 1024 / 1024, 2)) + 'M'
    if plain_size <= 1024 * 1024 * 1024 * 1024:
        return str(round(plain_size / 1024 / 1024 / 1024, 2)) + 'G'


def _getfile_time(file):
    """
    Get file time

    :param file: file_path+file_name
    :return: file time

    Examples:
        _getfile_time('E:/atome/git/data-tools/data_tools/third_features/create_table.py')
    """

    time_ = int(os.path.getmtime(file))
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_))


def get_file_list(file_path):
    """
    Get the filename,size and save_time of each file in the folder

    :param file_path:  The address of the folder to get the file information
    :return: dataframe with filename,size and save_time of each file

    Examples:
        get_file_list('E:/atome/git/data-tools/data_tools/tools/')
    """

    dir_list = os.listdir(file_path)
    if not dir_list:
        return None

    else:
        cfg = {}
        for filename in dir_list:
            cfg.update({filename: {'desc': {'save_time': _getfile_time(join(file_path, filename)),
                                            'size': _getfile_size(join(file_path, filename))}}})

        data = pd.DataFrame.from_dict(cfg, orient='index', dtype=None, columns=None).reset_index().rename(
            columns={'index': 'filename'})
        return data.merge(pd.json_normalize(data['desc'].map(lambda x: x), sep='_'),
                          left_index=True, right_index=True).drop(columns=['desc'], axis=1)
