import json
import re
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm
import category_encoders as ce

from ..gentools.advtools import cut_feas, WOE, JsonEncoder
from ..gentools.gentools import keep_df_value, extend_


def get_bins_bfwoe(df, fea_cfg, target, save_path=None):
    """
    Get bins according to fea_cfg, when save_path is not None, keep_value in fea_cfg do not support callback

    :param df: df
    :param fea_cfg: fea_cfg
    :param target: target
    :param save_path: save_path
    :return: dict

    Examples:
        cfg = pd.DataFrame({'feas':['normal_repay_amt_rf_due_max_p36d','normal_repay_amt_rf_due_max_p96d','ovd_days_rf_repay_max_p96d'],
                    'keep_value':[0,np.nan,[15,20]],
                    'tree_params':[{'max_depth': 4, 'min_samples_leaf': 0.05, 'max_bins':5},
                                   {'max_depth': 4, 'min_samples_leaf': 0.05, 'thred':2},
                                   {'max_depth': 2, 'min_samples_leaf': 0.02}]})
        bins_cfg = get_bins_bfwoe(x_train, fea_cfg=cfg.set_index('feas').to_dict("index"), target='p1_0', save_path='E:/huarui/')

        with open("E:/huarui/tree_bins",'r') as load_f:
            bins_cfg = json.load(load_f)
    """

    bins_list = []
    for key, value in tqdm(fea_cfg.items()):
        # get keep_value, tree_params
        keep_value = fea_cfg[key].get('keep_value', None)
        tree_params = fea_cfg[key].get('tree_params', {})
        # dropna
        data = df[[key] + [target]]
        # keep_value
        data = keep_df_value(data, key, keep_value)[1]
        bins = cut_feas(data, key, target, cut_params=tree_params,
                        thred=tree_params.get('thred', -1),
                        max_bins=tree_params.get('max_bins', 9999))
        bins_list.append(bins)

    # save
    out = pd.DataFrame(fea_cfg).T.reset_index().rename(columns={'index': 'feas'})
    out['bins'] = bins_list
    out = out.set_index('feas').to_dict("index")
    if save_path is not None:
        with open(join(save_path, 'tree_bins'), 'w') as json_file:
            json.dump(out, json_file, ensure_ascii=False, cls=JsonEncoder)

    return out


def woe_encode(df, woe_cfg, target, type_='calc', save_path=None):
    """
    Get bins and calc feas woe value according to woe_cfg

    :param df: df
    :param woe_cfg: woe_cfg
    :param target: target
    :param type_: calc/nocalc if calc we need to calc woe value according to woe_cfg else use woe value in woe_cfg
    :return:

    Examples:
        woe_cfg = {'normal_repay_amt_rf_due_max_p36d': {'keep_value': 0,
                                                         'tree_params': {'max_depth': 4, 'min_samples_leaf': 0.05, 'max_bins': 5},
                                                         'bins': [-np.inf,
                                                                  1112.8049926757812,
                                                                  1469.6749877929688,
                                                                  2600.364990234375,
                                                                  5703.169921875,
                                                                  np.inf]},
                    'normal_repay_amt_rf_due_max_p96d': {'keep_value': nan,
                                                         'tree_params': {'max_depth': 4, 'min_samples_leaf': 0.05, 'thred': 2},
                                                        'bins': [-np.inf,
                                                                 156.0999984741211,
                                                                 1112.8049926757812,
                                                                 1416.7150268554688,
                                                                 1636.8550415039062,
                                                                 2322.8800048828125,
                                                                 5408.929931640625,
                                                                 13832.5546875,
                                                                 np.inf]},
                    'ovd_days_rf_repay_max_p96d': {'keep_value': [15, 20],
                                                   'tree_params': {'max_depth': 2, 'min_samples_leaf': 0.02},
                                                   'bins': [-np.inf, 0.5, 1.5, np.inf]}}
        out, x_train = woe_encode(x_train, woe_cfg=woe_cfg, target='p1_0', type_='calc')
        out, x_test = woe_encode(x_test, woe_cfg=out, target='p1_0', type_='nocalc')
    """

    out = pd.DataFrame(woe_cfg).T.reset_index().rename(columns={'index': 'feas'})
    woe_value = []

    for key, value in tqdm(woe_cfg.items()):
        # keep value
        keep_value = value.get('keep_value', None)
        mask = keep_df_value(df, key, keep_value)[0]
        # cut
        df[f'{key}_afwoe'] = df[key]
        df.loc[mask, f'{key}_afwoe'] = pd.cut(df.loc[mask, key], value['bins'])
        df[f'{key}_afwoe'] = df[f'{key}_afwoe'].astype(str)
        # calc woe
        if type_ == 'calc':
            woe_value.append(WOE().woe_single_x(np.array(df[f'{key}_afwoe']), np.array(df[target]))[0])

    if type_ == 'calc':
        out['woe_cfg'] = woe_value
    out = out.set_index('feas').to_dict("index")
    df = df.replace(dict(zip([f'{c}_afwoe' for c in out.keys()], [c['woe_cfg'] for c in out.values()])))

    # save
    if save_path is not None:
        with open(join(save_path, 'nwoe_cfg'), 'w') as json_file:
            json.dump(out, json_file, ensure_ascii=False, cls=JsonEncoder)

    return out, df


def calc_flag(df, fea, val=None, flag='equal'):
    """
    Calculate flag from variable-flag map.

    :param df: df
    :param fea: fea
    :param val: val
    :param flag: fea flag
    :return:

    Examples:
        (1) calc_flag(x_train, 'ovd_days_rf_due_max_p96d', 2, 'clip_upper')
        (2) calc_flag(x_train, 'ovd_days_rf_due_max_p96d', 2, 'less_equal')
        (3) calc_flag(x_train, 'ovd_days_rf_due_max_p96d', 10, 'greater')
        (4) calc_flag(x_train, 'ovd_days_rf_due_max_p96d', np.nan, 'isnull')
    """

    if pd.isnull(val) or re.search('isnull', flag, re.I):
        return 1 * df[fea].isnull(), f'{fea}_isna'
    elif re.search('greater_equal', flag, re.I):
        return 1 * (df[fea] >= val), f'{fea}_ge{val}'
    elif re.search('greater', flag, re.I):
        return 1 * (df[fea] > val), f'{fea}_gt{val}'
    elif re.search('less_equal', flag, re.I):
        return 1 * (df[fea] <= val), f'{fea}_le{val}'
    elif re.search('less', flag, re.I):
        return 1 * (df[fea] < val), f'{fea}_lt{val}'
    elif re.search('clip_lower', flag, re.I):
        return df[fea].clip(lower=val), f'{fea}_clipL{val}'
    elif re.search('clip_upper', flag, re.I):
        return df[fea].clip(upper=val), f'{fea}_clipU{val}'
    elif re.search('equal', flag, re.I):
        return 1 * (df[fea] == val), f'{fea}_eq{val}'
    else:
        raise ValueError(f'unknown flag={flag}')


def add_flag(df, fea_cfg):
    """
    Calculate flag according to fea_cfg using func calc_flag and then add flag in df

    :param df: df
    :param fea_cfg: fea_cfg
    :return:

    Examples:
        cfg = pd.DataFrame({'feas':['ovd_days_rf_due_max_p96d','ely_gt0_repay_oid_cnt_rf_due_p96d','ely_gt0_repay_rto_rf_due_p1d'],
                    'keep_value':[np.nan,None,lambda x: x==0.5],
                    'flag':[[(0, 'equal'),(2, 'clip_upper'),(1,'less_equal')], [(10, 'greater_equal')], [(np.nan, 'isnull')]]})
        x_train = add_flag(x_train, fea_cfg=cfg.set_index('feas').to_dict('index'))
    """

    for fea, flag in tqdm(fea_cfg.items()):
        idx = keep_df_value(df, fea, drop_value=flag['keep_value'])[0]
        for val_flag in flag['flag']:
            srs, fea_new = calc_flag(df, fea, val_flag[0], flag=val_flag[1])
            df[fea_new] = srs
            df.loc[~idx, fea_new] = df.loc[~idx, fea]

    return df


def simple_transform_num(df, fea_cfg):
    """
    Transform numerical features using np.log,np.sqrt,np.square,np.power

    :param df:
    :param fea_cfg:
    :return:

    Examples:
        cfg = pd.DataFrame({'feas':['ovd_days_rf_due_max_p96d','ely_gt0_repay_oid_cnt_rf_due_p96d','ely_gt0_repay_rto_rf_due_p1d'],
                    'keep_value':[np.nan,10,lambda x: x==0.5],
                    'transform_func':[['log', 'sqrt', 'sqr', 'cube'], ['log'], [ 'sqr', 'cube']]})
        x_train = simple_transform_num(x_train, cfg.set_index('feas').to_dict("index"))
    """

    func_map = {'log': np.log,
                'sqrt': np.sqrt,
                'sqr': np.square,
                'cube': lambda x: np.power(x, 3)}

    for fea, value in fea_cfg.items():
        idx = keep_df_value(df, fea, drop_value=value['keep_value'])[0]
        for tf_func in value['transform_func']:
            if tf_func == 'log':
                df[f'{fea}_{tf_func}'] = func_map[tf_func](df[fea] + 0.00001)
            else:
                df[f'{fea}_{tf_func}'] = func_map[tf_func](df[fea])
            df.loc[~idx, f'{fea}_{tf_func}'] = df.loc[~idx, fea]

    return df


class Str2Number:

    @classmethod
    def str2number_fit(cls, df, fea_cfg, target, save_path=None):
        """
        Turn discrete variables into continuous variables
        when we use package category_encoders some parameters need to be explained:
            https://contrib.scikit-learn.org/category_encoders/
            handle_missing: str,   options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.
            handle_missing: str,   options are 'error', 'return_nan' and 'value', defaults to 'value', which returns the target mean.

            其实TargetEncoder、MEstimateEncoder和JamesSteinEncoder 三个的本质就是TS，只不过实际数据中有些群体可能太小或太不稳定而不可靠。
            许多有监督编码通过在组平均值和y的全局平均值之间选择一种中间方法来克服这个问题：wi*mean(y|x=i)+(1-wi)*mean(y)
            TargetEncoder、MEstimateEncoder和JamesSteinEncoder 根据它们定义wi的方式而有所不同。

            (1) ce.target_encoder.TargetEncoder:
                    For the case of categorical target: features are replaced with a blend of posterior probability of the target given particular categorical value
                    and the prior probability of the target over all the training data.（对于分类目标的情况：特征被替换为给定特定分类值的目标后验概率和目标在所有训练数据上的先验概率的混合）
                    For the case of continuous target: features are replaced with a blend of the expected value of the target given particular categorical value
                    and the expected value of the target over all the training data.（对于连续目标的情况：特征被替换为给定特定分类值的目标期望值和所有训练数据上目标的期望值的混合）
                    min_samples_leaf: default 1,  int,   minimum samples to take category average into account.
                    smoothing: default 1,  float,   smoothing effect to balance categorical average vs prior. Higher value means stronger regularization. The value must be strictly bigger than 0.
                    wi = 1/(1+np.exp(-(count_encoding-1)/smoothing)) 随着平滑度smoothing的增加，全局平均权值越来越多，导致正则化更强。

            (2) ce.m_estimate.MEstimateEncoder:
                    This is a simplified version of target encoder, which goes under names like m-probability estimate or additive smoothing with known incidence rates.
                    In comparison to target encoder, m-probability estimate has only one tunable parameter (m), while target encoder has two tunable parameters (min_samples_leaf and smoothing).
                    (这是目标编码器的简化版本，其名称为 m 概率估计或具有已知发生率的附加平滑。 与目标编码器相比，m 概率估计只有一个可调参数 (m)，而目标编码器有两个可调参数（min_samples_leaf 和平滑）)
                    randomized: default False,  bool,   adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                    sigma: default 0.05,  float,   standard deviation (spread or “width”) of the normal distribution.
                    m: default 1,  float,    this is the “m” in the m-probability estimate. Higher value of m results into stronger shrinking. M is non-negative.
                    wi = count_encoding/(count_encoding+m)  随着平滑度m的增加，全局平均权值越来越多，导致正则化更强。

            (3) ce.james_stein.JamesSteinEncoder:
                    model: str,    options are ‘pooled’, ‘beta’, ‘binary’ and ‘independent’, defaults to ‘independent’.
                    randomized: default False,  bool,   adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                    sigma: default 0.05,  float,    standard deviation (spread or “width”) of the normal distribution.
                    TargetEncoder和MEstimateEncoder既取决于组的数量，也取决于用户设置的参数值（分别是smoothing和m）。这不方便，因为设置这些权重是一项手动任务。
                    JamesSteinEncoder试图以一种基于统计数据的方式, 在不需要任何人为干预的情况下，设定一个最佳的工作环境。
                    y_mean, y_var = y.mean(), y.var()  方差
                    y_level_mean, y_level_var = x.replace(y.groupby(x).mean()), x.replace(y.groupby(x).var())
                    wi = 1-(y_level_var/(y_var+y_level_var)*(len(set(x))-3)/(len(set(x))-1))

            (4) ce.glmm.GLMMEncoder: （拟合y上的线性混合效应模型）
                    randomized: default False,  bool,    adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                    sigma: default 0.05,  float,    standard deviation (spread or “width”) of the normal distribution.
                    binomial_target: default None,  bool,    if True, the target must be binomial with values {0, 1} and Binomial mixed model is used.
                                                              If False, the target must be continuous and Linear mixed model is used.
                                                              If None (the default), a heuristic is applied to estimate the target type.
            (5) ce.woe.WOEEncoder:   （woe编码）
                    randomized: default False,  bool,    adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                    sigma: default 0.05,  float,    standard deviation (spread or “width”) of the normal distribution.
                    regularization: default 1,  float,    the purpose of regularization is mostly to prevent division by zero. When regularization is 0, you may encounter division by zero.

            (6) ce.leave_one_out.LeaveOneOutEncoder: （留一法编码）
                    sigma: default None,  float,    adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                                                    Sigma gives the standard deviation (spread or “width”) of the normal distribution.
                                                    The optimal value is commonly between 0.05 and 0.6. The default is to not add noise, but that leads to significantly suboptimal results.

            (7) ce.cat_boost.CatBoostEncoder:  （cat_boost 编码）
                    sigma: default None,  float,    adds normal (Gaussian) distribution noise into training data in order to decrease overfitting (testing data are untouched).
                                                    sigma gives the standard deviation (spread or “width”) of the normal distribution.
                    a: default 1,  float,    additive smoothing (it is the same variable as “m” in m-probability estimate). By default set to 1.

        :param df: df
        :param fea_cfg: fea_cfg
        :param target: target
        :param save_path: save_path
        :return:

        Examples:
            cfg = pd.DataFrame({
                    'feas':['ord_amt_max_in1d_is_holiday_p66d','LK_OUTPUT_MARITALSTATUS','province'],
                    'encoding_type':[{'selfcut':[[True],[False,'nan']]},
                                      {'TS':'TS'},
                                      {'TS':'TS',
                                       'ce.target_encoder.TargetEncoder':{'smoothing':2,'min_samples_leaf':50},
                                       'ce.m_estimate.MEstimateEncoder':{'sigma':0.1, 'm':1},
                                       'ce.james_stein.JamesSteinEncoder': {'sigma':0.1, 'randomized':True},
                                       'ce.glmm.GLMMEncoder': {'sigma':0.1, 'randomized':True, 'binomial_target':True},
                                       'ce.woe.WOEEncoder': {'sigma':0.1, 'regularization':1},
                                       'ce.leave_one_out.LeaveOneOutEncoder': {},
                                       'ce.cat_boost.CatBoostEncoder': {'sigma':0.1, 'a':1}}]})
            Str2Numbercfg = fp.Str2Number().str2number_fit(df, fea_cfg=cfg.set_index('feas').to_dict("index"), target=target, save_path='E:/huarui/')
        """

        str2intcfg = {}
        for fea, value in tqdm(fea_cfg.items()):
            df[fea] = df[fea].fillna('nan')
            str2intcfg_ = {}

            for func, params in value['encoding_type'].items():
                # when func is TS means we use standard TS Encode
                if func == 'TS':
                    str2intcfg_.update(
                        {'TS': df.groupby(fea, dropna=False)[target].mean().to_frame(fea).to_dict('dict')})
                # when func is selfcut means we cut str feas manual
                elif func == 'selfcut':
                    str2intcfg_.update({'selfcut': dict(zip(extend_(params),
                                                            extend_([[df[df[fea].isin(params[i:i + 1][0])][
                                                                          target].mean()] * len(params[i:i + 1][0])
                                                                     for i in range(len(params))])))})
                # when func startswith('ce') means we use package category_encoders
                elif func.startswith('ce'):
                    model = eval(func)(**params, handle_missing='value', handle_unknown='value') # handle_missing='return_nan', handle_unknown='return_nan'
                    _ = model.fit(df[fea], df[target])
                    str2intcfg_.update({func: model})
                else:
                    raise ValueError('func is non-compliant')

            str2intcfg.update({fea: [str2intcfg_]})
        if save_path is not None:
            pd.DataFrame({'feas': str2intcfg.keys(), 'str2numbercfg': str2intcfg.values()}). \
                to_pickle(join(save_path, 'str2numbercfg.pkl'))

        return str2intcfg

    @classmethod
    def str2number_transform(cls, df, str2intcfg):
        """
        Use after run fp.Str2Number().str2number_fit
        Turn discrete variables into continuous variables according to str2intcfg

        :param df: df
        :param str2intcfg: str2intcfg
        :return:

        Examples:
            tmp = pd.read_pickle('E:/huarui/str2numbercfg.pkl')
            str2numbercfg = dict(zip(list(tmp['feas']),list(tmp['str2numbercfg'])))
            x_train = fp.Str2Number().str2number_transform(x_train, str2numbercfg)
            x_test = fp.Str2Number().str2number_transform(x_test, str2numbercfg)

            # to plot feature bivar after str2number
            fplt.PlotUtils().plot_bivar(x_train, feas='province_JamesSteinEncoder', target=target, cut_params=None, title='feas',
                    yaxis='count', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red','black'])
            fplt.PlotUtils().plot_bivar(x_train, feas='province_CatBoostEncoder', target=target, cut_params=None, title='feas',
                    yaxis='count', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red','black'])
            fplt.PlotUtils().plot_bivar(x_train, feas='province_WOEEncoder', target=target, cut_params=None, title='feas',
                    yaxis='count', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red','black'])
        """

        for fea, value in str2intcfg.items():
            df[fea] = df[fea].fillna('nan')
            out = pd.DataFrame(index=range(df.shape[0]))

            for str2int_type, model in value[0].items():
                if str2int_type in ('TS', 'selfcut'):
                    out_ = df[[fea]].replace(model).rename(columns={fea: f'{fea}_{str2int_type}'})
                elif str2int_type.startswith('ce'):
                    out_ = model.transform(df[fea]).rename(columns={fea: f'{fea}_{str2int_type.split(".")[-1]}'})
                else:
                    raise ValueError('str2int_type is non-compliant')
                out = out.merge(out_, left_index=True, right_index=True)

            df = df.merge(out, left_index=True, right_index=True)

        return df

    #########
    # get woe_value after str2number_transform
    #########
    # cat_feas = [c for c in x_train if re.search('_TS$|_selfcut$|Encoder$',c)]
    # fea_cfg = dict(zip(cat_feas, [{'keep_value': np.nan, 'tree_params':
    #                     {'max_depth': 2, 'min_samples_leaf': 0.05, 'thred': 2}}] * len(cat_feas)))
    # bins_cfg = fp.get_bins_bfwoe(x_train, fea_cfg=fea_cfg, target=target, save_path=None)
    # out, x_train = fp.woe_encode(x_train, woe_cfg=bins_cfg, target=target, type_='calc')
    # out, x_test = fp.woe_encode(x_test, woe_cfg=out, target=target, type_='nocalc')

    ##########
    # to plot feature bivar after str2number
    ##########
    # fplt.PlotUtils().plot_bivar(x_train, feas='province_TS_afwoe', target=target, cut_params=None, title='feas',
    #                             yaxis='count', pyecharts=False, mark_line=True,
    #                             draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red', 'black'])
    # fplt.PlotUtils().plot_bivar(x_train, feas='province_CatBoostEncoder_afwoe', target=target, cut_params=None,
    #                             title='feas',
    #                             yaxis='count', pyecharts=False, mark_line=True,
    #                             draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red', 'black'])

    ##########
    # get cfg
    ##########
    # cfg = {}
    # _ = [cfg.update(x_train.groupby('_'.join(list(fea_cfg.keys())[i].split('_')[:-1]), dropna=False)[
    #                     f'{list(fea_cfg.keys())[i]}_afwoe']. \
    #                 max().sort_values().to_frame(f'{list(fea_cfg.keys())[i]}_afwoe').to_dict('dict')) for i in range(len(fea_cfg))]
    # cfg
