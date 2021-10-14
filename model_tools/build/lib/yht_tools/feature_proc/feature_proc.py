import json
import os
import re
from os.path import join, exists

import at.model_fitting as mf
import numpy as np
import pandas as pd

from ..ytools.ytools import eval_value, keep_df_value, WOE, JsonEncoder, cut_feas


def get_default_clipthre(df, feature, drop_value):
    '''
    use percentile to clip feas
    '''
    # dropna 
    data = df[[feature]]
    data = data.dropna(subset=[feature]).reset_index(drop=True)

    # drop_value
    data = keep_df_value(data, feature, drop_value)

    # get_default_clipthre
    Percentile = np.percentile(data[feature], [0, 25, 50, 75, 100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR * 1.5
    DownLimit = Percentile[1] - IQR * 1.5

    return {'lower': DownLimit, 'upper': UpLimit}


def get_clip_cfg(df, cfg, save_path):
    '''
    get clip cfg ref to 'cfg' & default_clipthre
    '''
    cfg_clip = cfg[pd.notnull(cfg['clip_params'])].reset_index(drop=True)
    feas_list = []
    clip_list = []
    keep_list = []

    for i in range(len(cfg_clip)):
        clip_params = eval_value(cfg_clip['clip_params'][i])
        feas = eval_value(cfg_clip['var'][i])
        drop_value = eval_value(cfg_clip['keep_value'][i])

        if clip_params == 'default':
            clip_params = get_default_clipthre(df, feature=feas, drop_value=drop_value)

        feas_list.append(feas)
        clip_list.append(clip_params)
        keep_list.append(drop_value)

    clip_cfg = pd.DataFrame({'var': feas_list, 'clip_params': clip_list, 'keep_value': keep_list})
    if save_path is not None:
        clip_cfg.to_csv(join(save_path, 'clip_cfg.csv'), index=False)

    return clip_cfg


def clip_feas(df, clip_cfg):
    '''
    clip feas by clip_cfg
    '''
    for i in range(len(clip_cfg)):
        feas = eval_value(clip_cfg['var'][i])
        clip_params = eval_value(clip_cfg['clip_params'][i])
        lower = clip_params.get('lower', -999999999)
        upper = clip_params.get('upper', 999999999)
        keep_value = eval_value(clip_cfg['keep_value'][i])

        df['{}_af_clip'.format(feas)] = df[feas]. \
            apply(lambda x: lower if x <= lower \
            else upper if x >= upper
        else x)

        if isinstance(keep_value, list):
            idx = df[feas].isin(keep_value)
        elif isinstance(keep_value, int) or isinstance(keep_value, float):
            idx = df[feas] == keep_value
        else:
            idx = pd.Series([False] * len(df))

        df.loc[idx, '{}_af_clip'.format(feas)] = df.loc[idx, feas]

    return df


def tree_fit(df, target, feature, keep_value, tree_params={'max_depth':4, 'thred':2}):
    '''
    fit tree
    '''
    # dropna 
    data = df[[feature] + [target]]
    data = data.dropna(subset=[feature]).reset_index(drop=True)

    # keep_value
    data = keep_df_value(data, feature, keep_value)

    # cut by tree
    bins = cut_feas(data, feature, target, cut_params=tree_params)[2]

    return feature, bins[1:-1]


def get_tree_bins(df, target, cfg, save_path):
    '''
    get_tree_bins by fit tree,tree params in cfg
    '''
    # save tree pic
    # if not exists(join(save_path, 'tree')):
        # os.makedirs(join(save_path, 'tree'))

    feas_list = []
    bins_list = []
    value_list = []
    cfg_tree = cfg[pd.notnull(cfg['tree_params'])].reset_index(drop=True)

    for i in range(len(cfg_tree)):
        feas = eval_value(cfg_tree['var'][i])
        tree_params = eval_value(cfg_tree['tree_params'][i])
        if tree_params == 'default':
            tree_params = {'max_depth': 4, 'min_samples_leaf': 0.06, 'thred': 2}
        keep_value = eval_value(cfg_tree['keep_value'][i])

        # start fit
        feas, bins = tree_fit(df, target, feas, keep_value, tree_params)
        feas_list.append(feas)
        bins_list.append(bins)
        value_list.append(keep_value)

    tree_bins = pd.DataFrame({'var': feas_list, 'bins': bins_list, 'keep_value': value_list})

    if save_path is not None:
        tree_bins.to_csv(join(save_path, 'tree_bins.csv'), index=False)

    return tree_bins


def cut_bytree(df, feature, bins, keep_value):
    '''
    cut_feas by bins
    '''
    df['{}_bin'.format(feature)] = pd.cut(df[feature], [-np.inf] + bins + [np.inf])
    df['{}_bin'.format(feature)] = df['{}_bin'.format(feature)].astype(str)

    if isinstance(keep_value, list):
        idx = df[feature].isin(keep_value)
    elif isinstance(keep_value, int) or isinstance(keep_value, float):
        idx = df[feature] == keep_value
    else:
        idx = pd.Series([False] * len(df))

    df.loc[idx, '{}_bin'.format(feature)] = df.loc[idx, feature]
    df['{}_bin'.format(feature)] = df['{}_bin'.format(feature)].astype(str)

    df['{}_bin'.format(feature)] = df['{}_bin'.format(feature)].fillna('na')
    # df[feature] = df[feature].fillna('na')

    return df


def get_woe_value(df, target, cols, dict_, suffix, save_path=None):
    '''
    get cwoe value
    '''
    for feas in cols:
        df['{}_{}'.format(feas, suffix)] = df[feas]

    if dict_ is None:
        dict_ = {}
        for feas in cols:
            dict_.update({'{}_{}'.format(feas, suffix):
                              WOE().woe_single_x(df[feas],
                                                 np.array(df[target]))[0]})

    df.replace(dict_, inplace=True)

    if save_path is not None:
        with open(join(save_path, 'woe_result'), 'w') as json_file:
            json.dump(dict_, json_file, ensure_ascii=False, cls=JsonEncoder)

    return df, dict_


def get_nwoe_by_tree(df, target, tree_bins, dict_=None, save_path=None, drop_cols=None):
    # cut feas
    for i in range(len(tree_bins)):
        df = cut_bytree(df, eval_value(tree_bins['var'][i]), \
                      eval_value(tree_bins['bins'][i]), \
                      eval_value(tree_bins['keep_value'][i]))

    cols = tree_bins['var'].tolist()
    cwoe_cols = [c + '_bin' for c in cols]

    # get_cwoe
    if dict_ is None:
        df, dict_ = get_woe_value(df, target, cwoe_cols, dict_=None,
                                  suffix='nwoe_bytree', save_path=save_path)
    else:
        df, dict_ = get_woe_value(df, target, cwoe_cols, dict_=dict_,
                                  suffix='nwoe_bytree', save_path=save_path)

    if drop_cols is not None:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

    return df, dict_


def str2int(df, feas_cfg, target, replace=True):
    """
    str feas to int
    """
    dict_out = {}
    for fea, str2int_cfg in feas_cfg.items():
        # 自定义分箱
        if isinstance(str2int_cfg, list):
            out = pd.DataFrame({fea: str2int_cfg})
            out['values'] = [df[df[fea].isin(out[i:i+1][fea].tolist()[0])][target].mean() for i in range(out.shape[0])]
            out = out.sort_values('values').reset_index(drop=True)
            out['values'] = range(out.shape[0])
            feas_list = []
            values_list = []
            for i in range(len(out)):
                feas_list.extend(out[fea][i])
                values_list.extend([i]*len(out[fea][i]))  
            dict_out.update({fea: dict(zip(feas_list, values_list))})

        # 特征字典化
        else:
            out = df.groupby([fea], dropna=False)[target].mean().sort_values()
            dict_out.update({fea:dict(zip(out.index,range(out.shape[0])))})

    # replace
    if replace is True:
        df = df.replace(dict_out)

    return df, dict_out


def calc_cwoe_by_tree(data, feas_cfg, target, save_path=None):
    """
    calc cwoe
    """

    ##############
    # str2int
    ##############
    df = data.copy()
    df, dict_out = str2int(df, feas_cfg, target, replace=True)

    ########
    # calc nwoe
    ########

    # get tree_split index
    need_ts_idx = list(np.where(np.array([isinstance(c,dict) for c in feas_cfg.values()])==True)[0])
    nneed_ts_idx = list(np.where(np.array([isinstance(c,dict) for c in feas_cfg.values()])==False)[0])

    # split by tree
    cfg = pd.DataFrame({'var':[list(feas_cfg.keys())[i] for i in need_ts_idx], 
                    'tree_params':[list(feas_cfg.values())[i] for i in need_ts_idx], 'keep_value':[[np.nan]]*len(need_ts_idx)})
    tree_bins = get_tree_bins(df, target, cfg, save_path=None)

    # no split by tree
    tree_bins_ = pd.DataFrame({'var':[list(feas_cfg.keys())[i] for i in nneed_ts_idx], 
                    'bins':[list(feas_cfg.values())[i] for i in nneed_ts_idx], 'keep_value':[[np.nan]]*len(nneed_ts_idx)})
    tree_bins_['bins'] = tree_bins_['bins'].apply(lambda x: [i for i in range(len(x))])

    #calc nwoe
    tree_bins = pd.concat([tree_bins,tree_bins_], axis=0).reset_index(drop=True)
    out = get_nwoe_by_tree(df[list(feas_cfg.keys())+[target]], target=target, tree_bins=tree_bins,
                             dict_=None, save_path=None, drop_cols=None)[0].drop(columns=target)

    # get cwoe_cfg
    cwoe_cfg = {}
    for feas in list(feas_cfg.keys()):
        tmp = pd.DataFrame.from_dict(dict_out[feas], orient='index').rename(columns={0:feas}).\
                reset_index().merge(out[[feas,f'{feas}_bin_nwoe_bytree']]).drop_duplicates().\
                drop(columns=feas).rename(columns={'index':feas, f'{feas}_bin_nwoe_bytree':f'{feas}_bin_cwoe_bytree'}).\
                sort_values(f'{feas}_bin_cwoe_bytree')
        cwoe_cfg.update({f'{feas}_bin_cwoe_bytree':dict(zip(tmp[feas],tmp[f'{feas}_bin_cwoe_bytree']))})

    # save
    if save_path is not None:
        with open(join(save_path, 'cwoe_result'), 'w') as json_file:
            json.dump(cwoe_cfg, json_file, ensure_ascii=False, cls=JsonEncoder)

    return cwoe_cfg


def get_cwoe(df, target, feas_cfg, otherfill_cfg, cwoe_cfg=None):
    """
    get_cwoe
    """
    if cwoe_cfg is None:
        cwoe_cfg = calc_cwoe_by_tree(df, feas_cfg, target)

    for fea in list(cwoe_cfg.keys()):
        df[fea] = df[re.sub('_bin_cwoe_bytree','',fea)]

    # replace cwoe
    replace_dict = {'replace':cwoe_cfg, 'other_fill':otherfill_cfg}
    df = df.replace(replace_dict['replace'])

    # otherfill
    for fea in list(cwoe_cfg.keys()):
        value = replace_dict['other_fill'][re.sub('_bin_cwoe_bytree','',fea)]
        if isinstance(value, str):
            value = eval(value)(list(cwoe_cfg[fea].values()))

        df[fea] = df[fea].apply(lambda x: value if isinstance(x,str) else x)

    return df, cwoe_cfg