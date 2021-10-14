import datetime as dt
import hashlib
import json
import math
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from gmssl import sm3, func
from pandas.api.types import is_string_dtype
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.cluster import KMeans
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


def cut_feas_bykmeans(data, feas, params={'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300}):
    """
    Aggregate feature value by KMeans

    :param data: dataframe
    :param feas: feas to aggregate value
    :param params: dict like {'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300}
    :return: aggregation boundary

    Examples:
        cut_feas_bykmeans(x_train, feas='ovd_days_rf_repay_last_p186d', params={'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300})
    """

    df = data[[feas]].sort_values(feas).reset_index(drop=True)
    df = df.dropna(subset=[feas]).reset_index(drop=True)

    kmeans = KMeans(n_clusters=params.get('n_clusters', 4),
                    init=params.get('init', 'k-means++'),  # Method for initialization k-means++/random
                    # Number of time the k-means algorithm will be run with different centroid seeds (k-means 算法将使用不同的质心种子运行的次数)
                    n_init=params.get('n_init', 10),
                    # Maximum number of iterations of the k-means algorithm for a single run(单次运行的 k-means 算法的最大迭代次数)
                    max_iter=params.get('max_iter', 300)
                    ).fit(df)

    df['AggregateByKMeans'] = kmeans.fit_predict(df)
    st = df.groupby(['AggregateByKMeans']).agg({feas: [min, max]})
    bins = sorted(list(st[feas]['max']))
    bins[len(bins) - 1] = np.inf
    bins = [-np.inf] + bins

    return bins


def cut_feas_bytree(data, feas, target, params={'max_depth': 3, 'min_samples_leaf': 0.05}):
    """
    Aggregate feature value by decision tree

    :param data: dataframe
    :param feas: str feas to aggregate value
    :param target: str target in decision tree
    :param params: dict like {'max_depth': 4, 'min_samples_leaf': 0.05}
    :return: aggregation boundary

    Examples:
        cut_feas_bytree(x_train, feas='ovd_days_rf_repay_last_p186d', target='p1_0', params={'max_depth': 3, 'min_samples_leaf': 0.02})
    """

    def tree_fit_bins(df, feas, target, max_depth, min_samples_leaf):
        model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model.fit(df[[feas]], df[target])

        left = model.tree_.children_left
        idx = np.argwhere(left != -1)[:, 0]
        bins = list(model.tree_.threshold[idx])
        bins = sorted(bins + [df[feas].min(), df[feas].max()])

        return bins

    try:
        df = data[[feas] + [target]].sort_values(feas).reset_index(drop=True)
        df = df.dropna(subset=[feas]).reset_index(drop=True)
        max_depth = params.get('max_depth', 3)
        min_samples_leaf = params.get('min_samples_leaf', 0.05)
        bins = tree_fit_bins(df, feas, target, max_depth, min_samples_leaf)

        if len(bins) == 2:
            for max_depth_ in np.arange(1, max_depth + 1, 1)[::-1]:
                for min_samples_leaf_ in np.arange(0.01, min_samples_leaf + 0.0000001, 0.01)[::-1]:
                    bins = tree_fit_bins(df, feas, target, max_depth_, min_samples_leaf_)
                    if len(bins) > 2:
                        # print(f'use more reasonable param: max_depth={max_depth_}, min_samples_leaf={min_samples_leaf_}')
                        return bins
    except KeyError:
        print('when use decision tree to aggregate feature, target can not be None')

    return bins


def cut_feas(data, feas, target, cut_params=20, thred=-1, max_bins=9999):
    """
    Aggregate feature value according to cut_params
    when cut_params dtype is list use cut
    when cut_params dtype is int use qcut
    when cut_params dtype is dict use KMeans or tree depending on cut_params.keys

    :param data: dataframe
    :param feas: feas to aggregate value
    :param target: str target in decision tree
    :param cut_params: support None/int/list/dict
    :param thred: when calc chi2_contingency result should>thred
    :param max_bins: max_bins
    :return: aggregation boundary

    Examples:
        (1) feas = 'ovd_days_rf_repay_last_p7d'
            ytls.cut_feas(x_train, feas, target=None, cut_params=None, thred=-1, max_bins=9999)
        (2) feas = 'ovd_days_rf_repay_last_p7d'
            ytls.cut_feas(x_train, feas, target='p1_0', cut_params=None, thred=2, max_bins=6)
            # warning when data[feas].nunique is too large, may get error
            # ValueError: The internally computed table of expected frequencies has a zero element at (0, 1).
        (3) feas = 'normal_repay_amt_rf_due_max_p96d'
            ytls.cut_feas(x_train, feas, target='p1_0', cut_params=10, thred=10, max_bins=9999)
        (4) feas = 'normal_repay_amt_rf_due_max_p96d'
            ytls.cut_feas(x_train, feas, target=None, cut_params=10, thred=-1, max_bins=9999)
        (5) feas = 'normal_repay_amt_rf_due_max_p96d'
            ytls.cut_feas(x_train, feas, target='p1_0', cut_params={'max_depth': 4, 'min_samples_leaf': 0.05}, thred=3, max_bins=5)
        (6) feas = 'normal_repay_amt_rf_due_max_p96d'
            ytls.cut_feas(x_train, feas, target='p1_0', cut_params={'n_clusters': 5, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300}, thred=3, max_bins=9999)
        (7) feas = 'normal_repay_amt_rf_due_max_p96d'
            ytls.cut_feas(x_train, feas, target=None, cut_params={'n_clusters': 3, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300}, thred=-1, max_bins=9999)
    """

    def merge_by_chi2(data, feas, target):
        """
        Calc the chi between each group using func chi2_contingency

        :param data: dataframe
        :param feas: feas to aggregate value
        :param target: target
        :return: groupby state, min_chi and state shape
        """

        df = data[[feas] + [target]].sort_values(feas).reset_index(drop=True)
        df[feas] = pd.cut(df[feas], bins)

        st = df.groupby([feas]).agg({feas: len, target: 'mean'}).rename(
            columns={feas: 'cnt'})  # do not need to dropna
        st['index'] = range(st.shape[0])
        st['good'] = st['cnt'] * (1 - st[target])
        st['bad'] = st['cnt'] * st[target]

        chi_list = []
        for i in range(len(st) - 1):
            try:
                chi_list.append(chi2_contingency(np.array(st[i:i + 2].iloc[:, -2:]))[0])
            except ValueError:
                chi_list.append(0)
        st['chi'] = chi_list + [np.nan]
        min_chi = st['chi'].min()

        return st, min_chi, st.shape

    # aggregate Continuous feature
    if is_string_dtype(data[feas]) is False:
        if cut_params is None:
            bins = sorted(data[pd.notnull(data[feas])][feas].unique().tolist())

        else:
            if isinstance(cut_params, list):
                bins = cut_params
            elif isinstance(cut_params, int):
                bins = pd.qcut(data[feas], cut_params, duplicates='drop', retbins=True)[1].tolist()
            else:
                try:
                    # aggregate feature value by KMeans
                    cut_params['n_clusters']
                    bins = cut_feas_bykmeans(data, feas, cut_params)
                except:
                    # aggregate feature value by decision tree
                    bins = cut_feas_bytree(data, feas, target, cut_params)

        if len(bins) <= 2:
            bins = [-np.inf, np.inf]
            thred, max_bins = -1, 9999
        else:
            bins[0], bins[len(bins) - 1] = -np.inf, np.inf

        if (target is not None):
            # get reasonable bins ref to thred and max_bins
            st, min_chi, st_shape = merge_by_chi2(data, feas, target)
            while (min_chi < thred) | (st_shape[0] > max_bins):
                try:
                    idx = st[st['chi'] == min_chi]['index'].values[0]
                    bins = bins[:idx + 1] + bins[idx + 2:]
                except IndexError:
                    bins = bins
                st, min_chi, st_shape = merge_by_chi2(data, feas, target)

    # aggregate Discrete feature
    else:
        bins = None

    return bins


class GetInformGain:
    """
    Calc information_gain & information_gain_rto
    """

    def get_entro(self, x):
        """
        Cala entropy

        :param x: np.array
        :return: entropy
        """

        entro = 0
        for x_value in set(x):
            p = np.sum(x == x_value) / len(x)
            if p != 0:
                logp = np.log2(p)
                entro -= p * logp
        return entro


    def get_condition_entro(self, x, y):
        """
        Calc condition entropy H(y|x)

        :param x: np.array
        :param y: np.array
        :return: condition entropy
        """

        condition_entro = 0
        for x_value in set(x):
            sub_y = y[x == x_value]
            temp_entro = self.get_entro(sub_y)
            condition_entro += (float(sub_y.shape[0]) / y.shape[0]) * temp_entro
        return condition_entro


    def get_information_gain(self, x, y):
        """
        Calc information gain

        :param x: np.array
        :param y: np.array
        :return: information gain
        """

        # calc information_gain
        inform_gain = (self.get_entro(y) - self.get_condition_entro(x, y))
        # calc information_gain_rto
        inform_gain_rto = inform_gain / self.get_entro(x)
        return [inform_gain, inform_gain_rto]


class WOE:
    """
    Calc woe & iv & sum(abs(woe_))
    """

    def __init__(self):
        self._WOE_MIN = -1.4
        self._WOE_MAX = 1.4


    def count_binary(self, a, event=1):
        """
        Calc the total number of good people and the total number of bad people

        :param a: np.array
        :param event: 1: means 1 is bad people; 0: means 0 is bad people
        :return: the total number of good people and the total number of bad people
        """

        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count


    def woe_single_x(self, x, y, event=1):
        """
        Calc woe & iv & sum(abs(woe_))

        :param x: np.array
        :param y: np.array
        :param event: 1: means 1 is bad people; 0: means 0 is bad people
        :return: iv & abs(woe)
        """

        event_total, non_event_total = self.count_binary(y, event=event)
        woe_dict, iv, woe_ = {}, 0, 0

        for x1 in pd.Series(x).unique():
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


class ExcelWrite:
    """
    Insert value/df/pic in excel
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
        Insert new_sheet in workbook

        :param workbook: workbook
        :param work_sheet_name: sheet_name
        :return:

        Examples:
            workbook = xlsxwriter.Workbook(join(report_save_path,'模型报告.xlsx'))
            workbook, worksheet = ytls.ExcelWrite().get_workbook_sheet(workbook, work_sheet_name='样本说明')
        """

        worksheet = workbook.add_worksheet(work_sheet_name)
        return workbook, worksheet


    def check_contain_chinese(self, check_str):
        """
        Determine whether the string contains Chinese

        :param check_str:  value
        :return: String contains Chinese situation

        Examples:
            check_contain_chinese('中国万岁！yes')
        """

        out = []
        for ch in str(check_str).encode('utf-8').decode('utf-8'):
            if u'\u4e00' <= ch <= u'\u9fff':
                out.append(True)
            else:
                out.append(False)
        return out, len(out) - sum(out), sum(out)


    def astype_insertvalue(self, value, decimal_point=4):
        """
        Astype insert_value

        :param value: insert value
        :param decimal_point: if value dtypes is float use decimal_point
        :return: insert value to table
        """

        if re.search('tuple|list|numpy.dtype|bool|str|numpy.ndarray|Interval|Categorical', str(type(value))):
            value = str(value)
        elif re.search('int', str(type(value))):
            value = value
        elif re.search('float', str(type(value))):
            value = round(float(value), decimal_point)
        else:
            value = 'nan'

        return value


    def insert_value2table(self, workbook, worksheet, value, insert_space, style={},
                           decimal_point=4, is_set_col=True):
        """
        Insert value in the table

        :param workbook: workbook
        :param worksheet: worksheet
        :param value: value
        :param insert_space: insert_space
        :param style: value style
        :param decimal_point: if value dtypes is float use decimal_point
        :param is_set_col: whether to set_column
        :return:

        Examples:
            (1) workbook, worksheet = insert_value2table(workbook, worksheet, value=40.121, insert_space='A1', style={},
                           decimal_point=4, is_set_col=True)
            (2) workbook, worksheet = insert_value2table(workbook, worksheet, value='CUSTOMERID', insert_space='A10', style={},
                           decimal_point=None, is_set_col=False)
        """

        # update style
        style_dict = self.orig_style
        style_dict.update(style)
        style = workbook.add_format(style_dict)

        # insert value
        value = self.astype_insertvalue(value, decimal_point=decimal_point)
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
        """
        Clac continuous_cnt

        :param list_:
        :param index_:
        :return:

        Examples:
            list_ = ['A','A','A','A','B','C','C','D','D','D']
            (1) calc_continuous_cnt(list_, 0) ===>('A', 0, 4)
            (2) calc_continuous_cnt(list_, 4) ===>('B', 4, 1)
            (3) calc_continuous_cnt(list_, 6) ===>('C', 6, 1)
        """

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
        """
        Merge dataframe index&column value

        :param workbook:
        :param worksheet:
        :param df:
        :param insert_space:
        :param style:
        :return:
        """

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
                    str_ = self.astype_insertvalue(str_)
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
                    str_ = self.astype_insertvalue(str_)
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


    def insert_df2table(self, workbook, worksheet, df, insert_space, style={}, is_merge=True):
        """
        Insert dataframe to sheet

        :param workbook: workbook
        :param worksheet: worksheet
        :param df: insert dataframe
        :param insert_space: insert_space
        :param style: df styple
        :param is_merge: whether to merge index&column value
        :return:

        Examples:
            st = x_train.groupby(['ord_succ_amt_sum_p186d_proc','p1_0']).agg({'p1_3':['count',sum]})
            (1) workbook, worksheet = insert_df2table(workbook, worksheet, df=st, insert_space='A1', style={}, is_merge=False)
            (2) workbook, worksheet = insert_df2table(workbook, worksheet, df=st, insert_space='A1', style={'bold':True}, is_merge=True)
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
                                                                  start_col) + j] + str(start_row + i), style=style_dict)
        # merge
        if is_merge is True:
            style_dict.update({'align': 'center'})
            workbook, worksheet = self.merge_df(workbook, worksheet, df, insert_space, style_dict)

        return workbook, worksheet


    def insert_pic2table(self, workbook, worksheet, pic_name, insert_space,
                         style={'x_scale': 0.5, 'y_scale': 0.5, 'x_offset': 0.2, 'y_offset': 0.2}):
        """
        Insert pic to table

        :param workbook: workbook
        :param worksheet: worksheet
        :param pic_name: pic save name
        :param insert_space: insert_space
        :param style: style
        :return: insert pic to table

        Examples:
            (1) workbook, worksheet = insert_pic2table(workbook, worksheet, pic_name='D:/一家人/孩子/宝宝照片/20210823(1).jpg', insert_space='A20')
        """
        _ = worksheet.insert_image(insert_space, pic_name, style)
        return workbook, worksheet


class ENCRYPT:

    @classmethod
    def func_md5(cls, value):
        """
        Get md5 code

        :param value: value
        :return: md5 code

        Examples:
            func_md5(value='杨海天')
        """

        m = hashlib.md5()
        m.update(str(value).strip().encode('utf-8'))
        return m.hexdigest()


    @classmethod
    def get_md5(cls, df, cols=['name', 'id_number', 'mobile_number']):
        """
        Do md5 encoding for specific columns

        :param df: dataframe
        :param cols: columns to encode md5
        :return:

        Examples:
            df = pd.DataFrame({'姓名':['张三','李四'], '手机号':[18817391693,18817391694], '身份证号':['32011219921218163X','320112199312181645']})
            get_md5(df, ['姓名','手机号','身份证号'])
        """

        for f in cols:
            df['{}_md5'.format(f)] = df[f].apply(lambda value: cls.func_md5(str(value)))
        return df


    @classmethod
    def jm_sha256_single(cls, value):
        """
        Get sha256 code

        :param value: value
        :return: sha256 code

        Examples:
            jm_sha256_single('张三')
        """

        hsobj = hashlib.sha256()
        hsobj.update(value.encode("utf-8"))
        return hsobj.hexdigest().upper()


    @classmethod
    def get_sha256(cls, df, cols=['name', 'id_number', 'mobile_number']):
        """
        Do sha256 encoding for specific columns

        :param df: dataframe
        :param cols: columns to encode sha256
        :return:

        Examples:
            df = pd.DataFrame({'姓名':['张三','李四'], '手机号':[18817391693,18817391694], '身份证号':['32011219921218163X','320112199312181645']})
            ENCRYPT().get_sha256(df, ['姓名','手机号','身份证号'])
        """

        for f in cols:
            df['{}_sha256'.format(f)] = df[f].apply(lambda value: cls.jm_sha256_single(str(value)))
        return df


    @classmethod
    def sm3_dt(cls, value):
        """
         Get sm3 code

        :param value: value
        :return: sm3 code

        Examples:
            sm3_dt(value='杨海天')
        """

        data = str(len(value)) + value
        result = sm3.sm3_hash(func.bytes_to_list(data.encode('utf-8')))
        return result


    @classmethod
    def get_sm3(cls, df, cols):
        """
        Do sm3 encoding for specific columns

        :param df: dataframe
        :param cols: columns to encode sm3
        :return:

        Examples:
            df = pd.DataFrame({'姓名':['张三','李四'], '手机号':[18817391693,18817391694], '身份证号':['32011219921218163X','320112199312181645']})
            ENCRYPT().get_sha256(df, ['姓名','手机号','身份证号'])
        """

        for f in cols:
            df['{}_sm3'.format(f)] = df[f].apply(lambda value: cls.sm3_dt(str(value)))
        return df




###########
# 需要重构代码
##########

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
