import math
import random
import re
from os.path import join

import at.model_fitting as mf
import at.utils.dumb_containers as dc
import catboost as ctb
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import xgboost as xgb
from catboost import Pool
from sklearn.model_selection import GroupKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

from ..ytools.ytools import cut_feas


# def xgb_apply(model, X_train, dummy=True):
# """
# params model: booster model
# params dummy
# """

# try:
# xgb_apply_feas = pd.DataFrame(model.apply(X_train))
# xgb_apply_feas.columns = ['tree{}'.format(c) for c in range(xgb_apply_feas.shape[1])]
# for f in list(xgb_apply_feas):
# xgb_apply_feas[f] = xgb_apply_feas[f].astype(str)

# if dummy is False:
# return xgb_apply_feas

# else:
# need = model.get_booster().trees_to_dataframe()
# idx = pd.isnull(need['Yes'])
# need = need[~idx].reset_index(drop=True)
# xgb_apply_feas = pd.get_dummies(xgb_apply_feas)

# def split_link(x, x_list):
# if x >= 0:
# x_list.append(x)
# return split_link(math.ceil(x / 2) - 1, x_list)
# else:
# return sorted(x_list)[:-1]

# tree_split_link = {}
# for c in range(2 ** (model.max_depth + 1) - 2):
# tree_split_link.update({c + 1: split_link(c + 1, [])})

# new_cols_list = []
# old_cols_list = []
# for tree in range(model.n_estimators):
# tmp = need[need['Tree'] == tree]
# for value in range(2 ** (model.max_depth + 1) - 2):
# new_cols_list.append(f'tree{tree}_node_{value + 1}_' + \
# '|'.join(
# list(set(tmp[tmp['Node'].isin(tree_split_link[value + 1])]['Feature']))))
# old_cols_list.append(f'tree{tree}_{value + 1}')

# xgb_apply_feas.rename(columns=dict(zip(old_cols_list, new_cols_list)), inplace=True)
# return xgb_apply_feas

# except:
# err = traceback.format_exc()
# print(err)


class GetModelApply:
    def __init__(self):
        pass

    @classmethod
    def model_apply(cls, model, df, dummy=True, cfg_save_path=None):
        """
        get tree model apply
        only xgb/lgb support
        """

        # get feasname
        feasname = get_model_feasname(model)

        ###########
        # apply
        ###########
        # xgboost model apply func
        if re.search('xgboost', str(model)):
            apply_feas = pd.DataFrame(model.predict(xgb.DMatrix(df[feasname]), pred_leaf=True))

        # lightgbm model apply func
        elif re.search('lightgbm', str(model)):
            apply_feas = pd.DataFrame(model.predict(df[feasname], pred_leaf=True))

        else:
            raise ValueError('only support xgb/lgb')

        apply_feas.columns = ['tree{}'.format(c) for c in range(apply_feas.shape[1])]
        for f in list(apply_feas):
            apply_feas[f] = apply_feas[f].astype(str)

        # get dummy
        if dummy is False:
            return apply_feas
        else:
            # get dummy
            apply_feas = pd.get_dummies(apply_feas)

            # rename columns
            if cfg_save_path is not None:
                new_cols_list, old_cols_list = cls._rename_tree_apply_feas(model)
                pd.DataFrame({'old_cols': old_cols_list, 'new_cols': new_cols_list}). \
                    to_csv(join(cfg_save_path, 'tree_apply_feas_cfg.csv'), index=False)
                apply_feas.rename(columns=dict(zip(old_cols_list, new_cols_list)), inplace=True)

            return apply_feas

    @classmethod
    def _rename_tree_apply_feas(cls, model):

        # xgboost model apply func
        if re.search('xgboost', str(model)):
            return cls._rename_xgbtree_apply_feas(model)

        # lightgbm model apply func
        elif re.search('lightgbm', str(model)):
            return cls._rename_lgbtree_apply_feas(model)

        else:
            raise ValueError('only support xgb/lgb')

    @classmethod
    def _rename_xgbtree_apply_feas(cls, model):

        # get trees_to_dataframe
        need = model.trees_to_dataframe()
        # get max_depth
        max_depth = math.ceil(math.log(need['Node'].max(), 2))

        idx = (pd.isnull(need['Yes'])) & (pd.isnull(need['No']))
        need = need[~idx].reset_index(drop=True)

        def get_xgb_node_feas(df, node, feas=[]):
            tmp = df[(df['Yes'] == node) | (df['No'] == node)]
            if tmp.shape[0] != 0:
                feas.append(list(tmp['Feature'])[0])
                get_xgb_node_feas(df, list(tmp['ID'])[0], feas)
            outfeas = list(set(feas))
            outfeas.sort(key=feas.index, reverse=True)
            return outfeas

        # rename
        new_cols_list = []
        old_cols_list = []
        for tree in sorted(need['Tree'].unique()):
            tmp = need[need['Tree'] == tree]
            for value in range(2 * 2 ** max_depth - 1):
                new_cols_list.append(f'tree{tree}_node_{value}_' + \
                                     '|'.join(get_xgb_node_feas(tmp, f'{tree}-{value}', [])))
                old_cols_list.append(f'tree{tree}_{value}')

        return new_cols_list, old_cols_list

    @classmethod
    def _rename_lgbtree_apply_feas(cls, model):

        # get trees_to_dataframe
        need = model.trees_to_dataframe()
        idx = (pd.isnull(need['left_child'])) & (pd.isnull(need['right_child']))
        need = need[~idx].reset_index(drop=True)

        def get_lgb_node_feas(df, node, feas=[]):
            tmp = df[(df['left_child'] == node) | (df['right_child'] == node)]
            if tmp.shape[0] != 0:
                feas.append(list(tmp['split_feature'])[0])
                get_lgb_node_feas(df, list(tmp['node_index'])[0], feas)
            outfeas = list(set(feas))
            outfeas.sort(key=feas.index, reverse=True)
            return outfeas

        new_cols_list = []
        old_cols_list = []
        for tree in sorted(need['tree_index'].unique()):
            tmp = need[need['tree_index'] == tree]
            for value in range(model.params['num_leaves']):
                new_cols_list.append(f'tree{tree}_node_{value}_' + \
                                     '|'.join(get_lgb_node_feas(tmp, f'{tree}-L{value}', [])))
                old_cols_list.append(f'tree{tree}_{value}')

        return new_cols_list, old_cols_list


def get_feature_names(model):
    if re.search('catboost', str(model)):
        return model.feature_names_

    elif re.search('xgboost', str(model)):
        return model.feature_names

    elif re.search('lightgbm', str(model)):
        return model.feature_name()

    elif re.search('statsmodels', str(model)):
        return list(model.params[:-1].index)

    else:
        raise ValueError('only support xgb/lgb/lr/cab')


class GetModelImportance:
    def __init__(self):
        pass

    @classmethod
    def _get_boster_shapimp(cls, model, X_train):

        """
        get shap importance
        params model: booster model
        params X_train: pd.DataFrame
        """

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        if isinstance(X_train, pd.DataFrame):
            max_display = len(X_train)
            feature_names = list(X_train)

        else:
            max_display = len(X_train.get_feature_names())
            feature_names = X_train.get_feature_names()

        global_shap_values = np.abs(shap_values).mean(0)

        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
        feature_inds = feature_order[:max_display]
        out = pd.DataFrame({'feature': [feature_names[i] for i in feature_inds], \
                            'shap_imp': global_shap_values[feature_inds]}).sort_values('shap_imp', ascending=False). \
            reset_index(drop=True)

        # if is_plot is True:
        # shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=out[out['shap_imp'] > 0].shape[0])

        return out

    @classmethod
    def _get_boster_imp(cls, model, X_train):

        """
        get boster importance
        """

        # catboost 计算特征速度太慢，目前用的是LossFunctionChange
        # http://www.atyun.com/42039.html
        ## PredictionValuesChange,LossFunctionChange,Interaction 是比较常用的
        if re.search('catboost', str(type(model))):
            data = model.get_feature_importance(type='LossFunctionChange',
                                                data=X_train, prettified=True)  # prettified to get dataframe
            data = data.fillna(0)
            return data.rename(columns={'Feature Id': 'feature', 'Importances': 'total_gain'})

        else:
            cols = list(X_train)
            data = pd.DataFrame(index=cols)

            if re.search('lightgbm', str(type(model))):  # lightgbm to get importance
                for importance_type in ['split', 'gain']:
                    tmp = pd.DataFrame(model.feature_importance(importance_type=importance_type)). \
                        rename(columns={0: importance_type}).rename(columns={'gain': 'total_gain'})
                    tmp.index = cols
                    data = data.merge(tmp, left_index=True, right_index=True, how='left')
                    data = data.fillna(0)

            elif re.search('xgboost', str(type(model))):  # xgb to get importance
                for importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                    try:  # use xgb.XGBRegressor
                        tmp = pd.DataFrame(model.get_booster().get_score(importance_type=importance_type), index=[0]). \
                            T.rename(columns={0: importance_type})
                    except AttributeError:
                        tmp = pd.DataFrame(model.get_score(importance_type=importance_type), index=[0]). \
                            T.rename(columns={0: importance_type})
                    data = data.merge(tmp, left_index=True, right_index=True, how='left')
                    data = data.fillna(0)

            else:
                raise ValueError('only support xgb/lgb/cab')

            ref_importance_type = 'total_gain'
            data = data.sort_values(ref_importance_type, ascending=False).reset_index().rename(
                columns={'index': 'feature'})
            # if is_plot is True:
            # fig, ax = plt.subplots(figsize=(13, 13))
            # xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=(data[ref_importance_type] > 0).sum(),
            # importance_type=ref_importance_type,
            # grid=False, show_values=False, xlabel=ref_importance_type)
            # plt.show()

            return data

    @classmethod
    def get_boster_imp(cls, model, X_train):

        shapimp = cls._get_boster_shapimp(model, X_train)
        imp = cls._get_boster_imp(model, X_train)
        imp = imp.merge(shapimp)
        return imp

    @classmethod
    def get_lr_imp(cls, model):

        imp = pd.DataFrame([(v[0], abs(float(v[3])))
                            for v in model.summary().tables[1].data[1:-1]], columns=['feature', 'zscore'])
        imp['importance'] = 1
        return imp

    @classmethod
    def get_permutation_imp(cls, model, x_test, feas, target, n_repeats=5):

        # calc base_auc & base_ks
        base_auc, base_ks, _, _ = SelectFeasByModel().evaluate_perf(model, x_test, feas, target)

        # shffle feas
        auc_add_mean, ks_add_mean, feas_list = [], [], []
        for fea in tqdm(feas):
            i, auc_add_list, ks_add_list = 0, [], []
            while i < n_repeats:
                tmp = x_test[feas + [target]]
                list_ = tmp[fea]
                random.shuffle(list_)
                tmp[fea] = list_

                # calc_auc,ks
                calc_auc, calc_ks, _, _ = SelectFeasByModel().evaluate_perf(model, tmp, feas, target)
                auc_add_list.append(base_auc - calc_auc)
                ks_add_list.append(base_ks - calc_ks)
                i = i + 1

            auc_add_mean.append(np.mean(auc_add_list))
            ks_add_mean.append(np.mean(ks_add_list))
            feas_list.append(fea)

        return pd.DataFrame(
            {'feature': feas_list, 'permutation_imp_auc': auc_add_mean, 'permutation_imp_ks': ks_add_mean}). \
            sort_values('permutation_imp_ks', ascending=False).reset_index(drop=True)

    @classmethod
    def get_model_imp(cls, model, X_train):

        if (re.search('xgboost', str(model))) or (re.search('lightgbm', str(model))) \
                or (re.search('catboost', str(model))):
            return cls.get_boster_imp(model, X_train)

        elif re.search('statsmodels', str(model)):
            return cls.get_lr_imp(model)

        else:
            raise ValueError('only support xgb/lgb/lr/cab')


class SelectFeasByModel():

    def __init__(self):
        pass

    @classmethod
    def _lr_fit(cls, x_train, feature, target):
        """
        lr model fit
        """
        feas = feature
        try:
            lm = sm.Logit(x_train[target], sm.add_constant(x_train[feas], prepend=False)).fit(disp=0)
            lm, feas = mf.stepwise_lr(x_train[feas + [target]],
                                      target, lm, prune_cnt=0, prune_low_zscore=True, step=1, thresh_pval=0.05)
            lm = sm.Logit(x_train[target], sm.add_constant(x_train[feas], prepend=False)).fit(disp=0)

        except:  # 共线性强的时候
            vif_check = x_train[feas]
            X = sm.add_constant(vif_check)
            vif_value = pd.Series([variance_inflation_factor(X.values, i)
                                   for i in range(X.shape[1])],
                                  index=X.columns)
            feas = list(set(vif_value[vif_value <= 10].index) - {'const'})  # drop 共线性模型

            lm = sm.Logit(x_train[target], sm.add_constant(x_train[feas], prepend=False)).fit(disp=0)
            lm, feas = mf.stepwise_lr(x_train[feas + [target]],
                                      target, lm, prune_cnt=0, prune_low_zscore=True, step=1, thresh_pval=0.05)
            lm = sm.Logit(x_train[target], sm.add_constant(x_train[feas], prepend=False)).fit(disp=0)

        return lm, feas, 0

    @classmethod
    def _xgb_fit(cls, x_train, x_test, feature, target, param=None):
        """
        xgb model fit
        """
        # get param
        # https://xgboost.readthedocs.io/en/latest/parameter.html#:~:text=XGBoost%20Parameters.%20%C2%B6.%20Before%20running%20XGBoost%2C%20we%20must,parameters%20depend%20on%20which%20booster%20you%20have%20chosen.
        if param is None:
            param = {
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
                'num_boost_round': 300,
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

        print(f'param is: {param}')
        result = {}
        # start fit
        feas = feature
        dtrain = xgb.DMatrix(x_train[feas], x_train[target])
        dtest = xgb.DMatrix(x_test[feas], x_test[target])
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(param, dtrain, num_boost_round=param.get('num_boost_round', 2000),
                          early_stopping_rounds=param.get('early_stopping_rounds', 50),
                          evals=watchlist, verbose_eval=param.get('verbose_eval', 100), evals_result=result)

        return model, feas, result

    @classmethod
    def _lgb_fit(cls, x_train, x_test, feature, target, param=None, cat_features=None):
        """
        lgb model fit
        """
        # get param 
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html
        if param is None:
            param = {'objective': 'regression',
                     'metric': ['auc'],
                     'boosting': 'dart',
                     # gbdt/rf/dart/goss(Gradient-based One-Side Sampling) 使用dart一般num_boost_round和learning_rate可以设的大一点,不太会过拟合
                     'num_boost_round': 300,
                     'learning_rate': 0.05,
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
                     'min_gain_to_split': 0.1,  # the minimal gain to perform split

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
                     'drop_rate': 0.1,
                     'max_drop': 2,
                     'skip_drop': 0.1,
                     'uniform_drop': False,  # set this to true, if you want to use uniform drop

                     # # only use when booster is goss
                     # 'top_rate': 0.2, # the retain ratio of large gradient data (单边梯度采样中，信息增益大的样本)
                     # 'other_rate': 0.1, # the retain ratio of small gradient data (单边梯度采样中，信息增益小的样本)

                     # # only use when tree_learner is voting
                     # 'top_k': 25, # 即选举并行中的K
                     }

        if cat_features is not None:
            x_train[cat_features] = x_train[cat_features].astype('category')
            x_test[cat_features] = x_test[cat_features].astype('category')

        print(f'param is: {param}')
        result = {}
        # start fit
        feas = feature
        dtrain = lgb.Dataset(x_train[feas], x_train[target])
        dtest = lgb.Dataset(x_test[feas], x_test[target], reference=dtrain)
        model = lgb.train(param, dtrain, verbose_eval=param.get('verbose_eval', 100),
                          num_boost_round=param.get('num_boost_round', 500),
                          valid_sets=[dtrain, dtest], valid_names=['train', 'test'],
                          evals_result=result)

        return model, feas, result

    @classmethod
    def _ctb_fit(cls, x_train, x_test, feature, target, param=None, cat_features=None):
        """
        catboost model fit
        """
        # get param
        # https://blog.csdn.net/mojir/article/details/94907968
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
        if param is None:
            param = {
                'loss_function': 'Logloss',
                'custom_loss': 'AUC',
                'eval_metric': 'AUC',
                'iterations': 500,
                'learning_rate': 0.03,
                'l2_leaf_reg': 10,
                'random_strength': 2,
                # The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.

                'use_best_model': True,
                'max_depth': 4,
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

        if cat_features is not None:
            x_train[cat_features] = x_train[cat_features].astype('str')
            x_test[cat_features] = x_test[cat_features].astype('str')

        print(f'param is: {param}')
        # start fit
        feas = feature
        dtrain = Pool(data=x_train[feas], label=x_train[target], cat_features=cat_features)
        dtest = Pool(data=x_test[feas], label=x_test[target], cat_features=cat_features)
        model = ctb.train(dtrain=dtrain, params=param, iterations=param.get('iterations', 500),
                          eval_set=[dtrain, dtest], verbose=param.get('verbose_eval', 100),
                          early_stopping_rounds=param.get('od_wait', 50), plot=False)

        results = model.get_evals_result()
        result = {}
        result.update({'train': {'auc': results['validation_0']['AUC']}})
        result.update({'test': {'auc': results['validation_1']['AUC']}})

        return model, feas, result

    @classmethod
    def model_fit(cls, model_type, x_train, feature, target,
                  x_test=None, param=None, cat_features=None):
        """
        xgb/lgb/cab/lr model_fit
        """

        if model_type == 'xgb':
            return cls._xgb_fit(x_train, x_test, feature, target, param)

        elif model_type == 'lgb':
            return cls._lgb_fit(x_train, x_test, feature, target, param, cat_features)

        elif model_type == 'lr':
            return cls._lr_fit(x_train, feature, target)

        elif model_type == 'cab':
            return cls._ctb_fit(x_train, x_test, feature, target, param, cat_features)

        else:
            raise ValueError('only support xgb/lgb/cab/lr')

    @classmethod
    def evaluate_perf(cls, model, X_train, feas, target, save_path=None, title=None):
        """
        xgb/lgb/lr model evaluate_perf
        """

        if re.search('xgboost', str(model)):
            ks_, auc_, pic, show_list = dc.evaluate_performance(X_train[target].values,
                                                                model.predict(
                                                                    xgb.DMatrix(X_train[feas], X_train[target])))

        elif re.search('lightgbm', str(model)):
            ks_, auc_, pic, show_list = dc.evaluate_performance(X_train[target].values,
                                                                model.predict(X_train[feas]))

        elif re.search('statsmodels', str(model)):
            ks_, auc_, pic, show_list = dc.evaluate_performance(X_train[target].values,
                                                                model.predict(sm.add_constant(X_train[feas],
                                                                                              prepend=False,
                                                                                              has_constant='add')))

        elif re.search('catboost', str(model)):
            ks_, auc_, pic, show_list = dc.evaluate_performance(X_train[target].values,
                                                                [c[1] for c in model.predict(X_train[feas],
                                                                                             prediction_type='Probability')])

        else:
            raise ValueError('only support xgb/lgb/lr/cab')

        if save_path is not None:
            pic.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')

        return ks_, auc_, pic, show_list

    @classmethod
    def _model_cv(cls, num_folds, param, x_train, feas,
                  target, model_type, group_feas='OBJECTNO',
                  cat_features=None):
        """
        get cv
        """
        if group_feas is None:
            x_train = x_train.reset_index()
            group_feas = 'index'

        k_fold = GroupKFold(n_splits=num_folds)
        train_x = x_train[feas + [target]].copy()
        train_y = x_train[target].copy()
        kf = k_fold.split(train_x, train_y, groups=x_train[group_feas])  # split trainx,trainy
        model_dict = {}

        # start cv
        for i, (train_fold, validate) in enumerate(kf):
            print('*' * 60 + ' item {} '.format(i) + '*' * 60)
            X_train, X_validate = train_x.loc[train_fold], train_x.loc[validate]
            print(X_train.shape, X_validate.shape)
            model_, feas_, _ = cls.model_fit(model_type, X_train, feas, target, X_validate, param, cat_features)
            model_dict.update({i: [model_, feas_]})

        x_train.drop(columns='index', axis=1, errors='ignore', inplace=True)
        return model_dict

    @classmethod
    def _model_cv_perfm(cls, model_dict, data_dict, target, metric='ks'):

        output = None
        ## get ks,auc ##
        for perf_name, perf_value in data_dict.items():
            tmp = pd.DataFrame()

            for model, feas in model_dict.values():
                ks_, auc_, _, _ = cls.evaluate_perf(model, perf_value, feas, target)
                tmp_ = pd.DataFrame({f'{perf_name}_ks': [ks_], f'{perf_name}_auc': [auc_]})
                tmp = pd.concat([tmp, tmp_], axis=0).reset_index(drop=True)

            if output is None:
                output = tmp.copy()
            else:
                output = output.merge(tmp, left_index=True, right_index=True)

        ## calc metric volatility
        sum_ = 0
        for i in range(len(data_dict)):
            for j in range(i + 1, len(data_dict)):
                sum_ = sum_ + abs(
                    output[f'{list(data_dict.keys())[i]}_{metric}'] - output[f'{list(data_dict.keys())[j]}_{metric}'])
        output[f'{metric}_volatility'] = sum_

        return output

    @classmethod
    def model_cv_result(cls, num_folds, param, x_train, feas,
                        target, model_type, data_dict,
                        metric='ks', group_feas='OBJECTNO', cat_features=None):

        model_dict = cls._model_cv(num_folds, param, x_train, feas,
                                   target, model_type, group_feas, cat_features)

        return model_dict, cls._model_cv_perfm(model_dict, data_dict, target, metric)

    @classmethod
    def cv_feature_imp(cls, X_train, model_dict, need_index, importance_type='booster_imp'):
        """
        # importance_type: booster_imp or shap_imp
        """
        imp = None

        for item, (model, feas) in enumerate(model_dict.values()):
            # model is xgb/lgb/cab
            if (re.search('xgboost', str(model))) or (re.search('lightgbm', str(model))) \
                    or (re.search('catboost', str(model))):

                if importance_type == 'booster_imp':
                    imp_ = GetModelImportance()._get_boster_imp(model, X_train)
                    imp_ = imp_[['feature', 'total_gain']].rename(columns={'total_gain': f'importance_{item}'})
                else:
                    imp_ = GetModelImportance()._get_boster_shapimp(model, X_train)
                    imp_ = imp_[['feature', 'shap_imp']].rename(columns={'shap_imp': f'importance_{item}'})

            elif re.search('statsmodels', str(model)):
                # model is lm
                imp_ = GetModelImportance().get_lr_imp(model)
                imp_ = imp_[['feature', 'importance']].rename(columns={'importance': f'importance_{item}'})

            else:
                raise ValueError('only support xgb/lgb/lr')

            if imp is None:
                imp = imp_
            else:
                imp = imp.merge(imp_, how='outer')
        imp = imp.fillna(0)

        sum_ = 0
        for idx in list(need_index):
            sum_ = sum_ + imp[f'importance_{idx}']

        imp['importance_all_need'] = sum_

        return imp, imp[['feature', 'importance_all_need']].sort_values('importance_all_need', ascending=False)


def get_model_feasname(model):
    """
    get model feasname
    """
    # lr model
    if re.search('statsmode', str(model)):
        return list(model.params.index)[:-1]

    # xgb model
    elif re.search('xgboost', str(model)):
        return model.feature_names

        # lgb model
    elif re.search('lightgbm', str(model)):
        return model.feature_name()

        # ctb model
    elif re.search('catboost', str(model)):
        return model.feature_names_

    else:
        raise ValueError('only support xgb/lgb/lr/cab')


def get_model_predict(df, model, score_name='pred'):
    # get model_feas
    feasname = get_model_feasname(model)

    # predict
    if re.search('statsmode', str(model)):
        score_list = list(model.predict(sm.add_constant(df[feasname], prepend=False, has_constant='add')))

    # xgb model
    elif re.search('xgboost', str(model)):
        score_list = list(model.predict(xgb.DMatrix(df[feasname])))

    # lgb model
    elif re.search('lightgbm', str(model)):
        score_list = list(model.predict(df[feasname]))

    # ctb model
    elif re.search('catboost', str(model)):
        score_list = [c[1] for c in model.predict(df[feasname], prediction_type='Probability')]

    else:
        raise ValueError('only support xgb/lgb/lr/cab')

    df[score_name] = score_list

    return score_list, df


def get_pred_cutoff(data, score, target, cut_params, ispred=False):
    """
    get score cut_off
    """
    woe_max = 5
    _, st, bins = cut_feas(data, score, target, cut_params=cut_params)
    st.rename(columns={score: '总样本数', target: '坏样本占比'}, inplace=True)

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

    # get woe/iv
    st['pyi'] = st['坏样本数'] / list(st.tail(1)['累计坏样本数'])[0]
    st['pni'] = st['好样本数'] / list(st.tail(1)['累计好样本数'])[0]
    st['woe'] = np.log(st['pyi'] / st['pni'])
    st.loc[st['坏样本占比'] == 1, 'woe'] = woe_max
    st.loc[st['坏样本占比'] == 0, 'woe'] = -woe_max
    st['iv'] = st['woe'] * (st['pyi'] - st['pni'])
    st['iv'] = st['iv'].sum()
    st.drop(columns=['pyi', 'pni'], inplace=True)
    st['lift'] = st['坏样本占比'] / data[target].mean()

    if ispred is True:
        st['预测坏样本概率'] = [data[(bins[i] < data[score]) & (data[score] <= bins[i + 1])][score].mean() for i in
                         range(len(bins) - 1)]

    st_show = st.copy()
    for c in [c for c in st_show if re.search('占比|ks', c)]:
        st_show[c] = st_show[c].apply(lambda x: f'{round(100 * x, 2)}%')

    return st, st_show, bins
