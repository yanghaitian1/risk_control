import math
import re
from os.path import join

import at.plot as atp
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from graphviz import Source
from ipywidgets import Layout
from ipywidgets import widgets
from pandas.api.types import is_string_dtype
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from sklearn.linear_model import LinearRegression

from ..ytools.ytools import cut_feas


class TabOut:
    def __init__(self):
        pass

    @classmethod
    def display_(cls, value):
        try:
            display.display(value.render_notebook())
        except AttributeError:
            if isinstance(value, matplotlib.figure.Figure) or isinstance(value, pd.DataFrame):
                display.display(value)
            elif isinstance(value, list):
                for value_ in value:
                    print(value_)
            else:
                print(value)

    @classmethod
    def tab_output_(cls, description, value, output, width='333px'):
        """
        get button & output
        """
        button = widgets.Button(description=f"{description}",
                                layout=Layout(width=width, height='40px'),  # icon='check' # check是加了一个勾
                                )

        def showOutput(btn):
            output.clear_output()
            with output:
                if isinstance(value, list):
                    for value_ in value:
                        cls.display_(value_)
                else:
                    cls.display_(value)

        button.on_click(showOutput)
        ui = widgets.HBox([button])

        return ui, output

    @classmethod
    def get_tab_out(cls, params, box_style='info', column_cnt=5):
        output = widgets.Output()
        ui_list = []
        min_cnt = min(len(params), column_cnt)
        width = f'{int(1000 / min_cnt)}px'

        for keys, values in params.items():
            ui, output = cls.tab_output_(keys, values, output, width=width)
            ui_list.append(ui)

        ui_list2 = []
        for i in range(len(ui_list) // column_cnt + 1):
            ui_list2.append(widgets.HBox(ui_list[i * column_cnt:(i + 1) * column_cnt]))

        return widgets.VBox(ui_list2 + [output], box_style=box_style)


# def plot_bivar_feas(data, feas, target, cut_params, title='feas', draw_lin=False,
# yaxis='count', pycharts=False, save_path=None):
# """
# plot_bivar_feas
# """

# if is_string_dtype(data[feas]):
# st = data.groupby(feas, dropna=False).agg({feas: len, target: 'mean'})
# else:
# df, st, bins = cut_feas(data, feas, target, cut_params)

# st['rto'] = st[feas] / data.shape[0]
# st['mean'] = data[target].mean()
# st.rename(columns={feas: title}, inplace=True)

# if draw_lin is True:
# idx = st.index != 'nan'
# st_tmp = st[idx]
# st_tmp['index'] = range(st_tmp.shape[0])
# lin_reg = LinearRegression().fit(st_tmp['index'].values.reshape(-1, 1), st_tmp[target].values.reshape(-1, 1))
# col_name = f'coef={round(lin_reg.coef_[0][0], 6)}, intercept={round(lin_reg.intercept_[0], 6)}'
# st_tmp[col_name] = lin_reg.coef_[0] * list(range(st_tmp.shape[0])) + lin_reg.intercept_[0]
# st = st.merge(st_tmp[['index', col_name]], left_index=True, right_index=True, how='left')

# if pycharts is False:
# x = st.index.tolist()
# fig = plt.figure(figsize=[20, 10])
# ax1 = fig.add_subplot(111)
# if yaxis == 'count':
# plt_bar = ax1.bar(x, st[title].tolist(), color="#1C86EE", label='feas cnt')
# elif yaxis == 'mean':
# plt_bar = ax1.bar(x, st['rto'].tolist(), color="#1C86EE", label='feas rto')
# else:
# raise ValueError('wrong yaxis')
# ax = plt.gca()  # 设置x轴刻度的数量
# ax.locator_params("x")
# ax.tick_params(labelsize=20, rotation=90)

# ax2 = plt.twinx()  # 添加坐标轴
# plt_target = ax2.plot(x, st[target].tolist(), color="red", linewidth=1.5, label=f'{target} rto')
# ax2.tick_params(labelsize=20)
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)

# try:
# plt_line = ax2.plot(x, st[col_name], color="black", linewidth=3, label=f'{col_name}')
# ax.set_title(f'{title}\n\n {col_name}', fontsize=30)
# except UnboundLocalError:
# plt_line = ax2.plot(x, st['mean'], color="black", linewidth=1, label='mean')
# ax.set_title(f'{title}', fontsize=30)

# xtick_chunk = 10
# if len(x) > 25:
# plt.xticks([int(len(x) / xtick_chunk) * i for i in range(xtick_chunk)] + [len(x) - 1])
# if save_path is not None:
# plt.savefig(join(save_path, f'{title}.png'))

# plt.close(fig)
# return fig

# else:
# try:
# col_name
# except UnboundLocalError:
# col_name = 'mean'

# if yaxis == 'count':
# return atp.PlotUtils.plot_twinx(st[title], st[[target, col_name]],
# mark_line=[], mark_point=None, title=f'{title}')
# else:
# return atp.PlotUtils.plot_twinx(st['rto'], st[[target, col_name]],
# mark_line=[], mark_point=None, title=f'{title}')


def _get_state(data, feas, target, cut_params, title='feas'):
    """
    get feas state
    """
    ## cut cat feas ##
    if is_string_dtype(data[feas]):
        if target is None:
            st = data.groupby(feas, dropna=False).agg({feas: len})
        else:
            st = data.groupby(feas, dropna=False).agg({feas: len, target: 'mean'})

    ## cut num feas ##
    else:
        df, st, bins = cut_feas(data, feas, target, cut_params)

    if target is not None:
        st['mean'] = data[target].mean()
    st['rto'] = st[feas] / data.shape[0]
    st.rename(columns={feas: title}, inplace=True)

    return st


def _plot(state, plot_type='bar', yaxis='count', title='feas', pycharts=False, label=None):
    ## get x_label ##
    st = state.copy()
    st.rename(columns={list(st)[0]: title}, inplace=True)
    x = st.index.tolist()

    if label is None:
        label = 'cnt'
        ## get yaxis ##
        if yaxis == 'mean':
            st[title] = st['rto']
            label = 'rto'
    else:
        if yaxis == 'mean':
            st[title] = st['rto']

            ## draw bar ##
    if pycharts is False:
        fig = plt.figure(figsize=[20, 10])
        if plot_type == 'bar':
            plt_bar = plt.bar(x, st[title].tolist(), color="#1C86EE", label=label)

        elif plot_type == 'barh':
            plt_bar = plt.barh(x, st[title].tolist(), color="#1C86EE", label=label)

        elif plot_type == 'line':
            plt_line = plt.plot(x, st[title].tolist(), color="red", label=label)
        else:
            raise ValueError('wrong plot_type')

        plt.legend(loc='upper right', fontsize=20)
        plt.tick_params(labelsize=20)
        plt.xticks(rotation=90)
        plt.title(f'{title}', fontsize=30)
        xtick_chunk = 10
        if len(x) > 25:
            plt.xticks([int(len(x) / xtick_chunk) * i for i in range(xtick_chunk)] + [len(x) - 1])

        plt.close(fig)
        return fig

    else:
        st[label] = st[title]
        return atp.PlotUtils.pe_plot(st[label], title=title, kind=plot_type, mark_line=None,
                                     mark_point=None, theme='white')


def _plot_bivar(state, target, draw_lin=False, yaxis='count', title='feas',
                pycharts=False, save_path=None):
    ## get x_label ##
    st = state.copy()
    st.rename(columns={list(st)[0]: title}, inplace=True)
    x = st.index.tolist()

    label = 'cnt'
    ## get yaxis ##
    if yaxis == 'mean':
        st[title] = st['rto']
        label = 'rto'

    ## get coef&intercept
    if draw_lin is True:
        idx = st.index != 'nan'
        st_tmp = st[idx]
        st_tmp['index'] = range(st_tmp.shape[0])
        lin_reg = LinearRegression().fit(st_tmp['index'].values.reshape(-1, 1), st_tmp[target].values.reshape(-1, 1))
        col_name = f'coef={round(lin_reg.coef_[0][0], 6)}, intercept={round(lin_reg.intercept_[0], 6)}'
        st_tmp[col_name] = lin_reg.coef_[0] * list(range(st_tmp.shape[0])) + lin_reg.intercept_[0]
        st = st.merge(st_tmp[['index', col_name]], left_index=True, right_index=True, how='left')

    ## get bivar ## 
    if pycharts is False:
        x = st.index.tolist()
        fig = plt.figure(figsize=[20, 10])
        ax1 = fig.add_subplot(111)
        plt_bar = ax1.bar(x, st[title].tolist(), color="#1C86EE", label=label)

        ax = plt.gca()  # 设置x轴刻度的数量
        ax.locator_params("x")
        ax.tick_params(labelsize=20, rotation=90)

        ax2 = plt.twinx()  # 添加坐标轴
        plt_target = ax2.plot(x, st[target].tolist(), color="red", linewidth=1.5, label=target)
        ax2.tick_params(labelsize=20)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)

        try:
            plt_line = ax2.plot(x, st[col_name], color="black", linewidth=3, label=f'{col_name}')
            ax.set_title(f'{title}\n\n {col_name}', fontsize=30)
        except UnboundLocalError:
            plt_line = ax2.plot(x, st['mean'], color="black", linewidth=1, label='mean')
            ax.set_title(f'{title}', fontsize=30)

        xtick_chunk = 10
        if len(x) > 25:
            plt.xticks([int(len(x) / xtick_chunk) * i for i in range(xtick_chunk)] + [len(x) - 1])
        if save_path is not None:
            plt.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')

        plt.close(fig)
        return fig


    else:
        try:
            col_name
        except UnboundLocalError:
            col_name = 'mean'

        st[label] = st[title]
        return atp.PlotUtils.plot_twinx(st[label], st[[target, col_name]],
                                        mark_line=[], mark_point=None, title=title)


def plot_bivar_feas(data, feas, target, cut_params, title='feas',
                    plot_type='bar', yaxis='count', pycharts=False,
                    draw_lin=False, save_path=None):
    ## get state ##
    st = _get_state(data, feas, target, cut_params, title)

    ## plot ##
    if target is None:
        if pycharts is True:
            return _plot(st, plot_type, yaxis, title, pycharts=True)
        else:
            out = _plot(st, plot_type, yaxis, title, pycharts=False)
            if save_path is not None:
                out.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')
            return out

    else:
        if pycharts is True:
            return _plot_bivar(st, target, draw_lin, yaxis, title,
                               pycharts=True, save_path=save_path)
        else:
            out = _plot_bivar(st, target, draw_lin, yaxis, title,
                              pycharts=False, save_path=save_path)
            if save_path is not None:
                out.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')
            return out


def _get_twix_state(compare_dict, feas, target, cut_params, title='feas'):
    """
    get_twix_state
    params compare_dict: like {'x_train':x_train,
                               'x_test':x_test,
                               'oot':oot}
    params cut_params: like 10,[1,2,3],{'max_depth': 4, 'thred': 2},{'KMeans': 4}
    """
    # get compare_list & compare_name
    compare_list = list(compare_dict.values())
    compare_name = list(compare_dict.keys())

    # get bins 
    if cut_params is None:
        bins = None
    else:
        bins = cut_feas(compare_list[0], feas, target, cut_params)[2]

    # get state
    for i in range(len(compare_list)):
        name = compare_name[i]
        tmp = _get_state(compare_list[i], feas, target, cut_params=bins, title=title). \
            rename(columns={title: f'{name}_{title}', target: f'{name}_{target}', 'mean': f'{name}_mean',
                            'rto': f'{name}_rto'}). \
            reset_index()
        if i == 0:
            st = tmp.copy()
        else:
            st = st.merge(tmp, how='outer')

    st = st.fillna(0)
    try:
        st['index']
    except KeyError:
        st['index'] = st[feas]

    st = pd.concat([st[st['index'] != 'nan'], st[st['index'] == 'nan']], axis=0).reset_index(drop=True)

    return st


def _plot_twinx_withnotarget(st, compare_name, yaxis, title, pycharts=False):
    """
    plot_twinx_withnotarget
    """
    # bar is 'count' or 'mean'
    if yaxis == 'count':
        feas_name = title
    else:
        feas_name = 'rto'

    # plt
    if pycharts is False:
        # get x label
        length = len(compare_name)
        bar_width = 0.9 / length
        x = np.array(range(len(st)))
        tick_label = st['index']

        # plot
        fig = plt.figure(figsize=[20, 10])
        ax1 = fig.add_subplot(111)

        # draw bar
        for i, name in enumerate(compare_name):
            ax1.bar([c + i * bar_width for c in x], st[f'{name}_{feas_name}'].tolist(), bar_width, alpha=0.5,
                    label=f'{name}')
        ax = plt.gca()  # 设置x轴刻度的数量
        ax.locator_params("x")
        ax.tick_params(labelsize=20, rotation=90)
        x_ = x + length * bar_width / length
        xtick_chunk = 10
        if len(x_) > 50:
            x_ = [int(len(x_) / xtick_chunk) * i for i in range(xtick_chunk)] + [len(x_) - 1]
            tick_label = [tick_label[i] for i in x_]

        # get legend
        plt.legend(fontsize=20)
        plt.title(f'{title}', fontsize=30)
        plt.xticks(x_, tick_label)
        plt.close(fig)

        return fig

    # ptecharts plot
    else:
        need = st.set_index('index')[[c for c in st if re.search(feas_name, c)]]
        need.rename(columns={f'{name}_{feas_name}': f'{name}' for name in compare_name}, inplace=True)

        return atp.PlotUtils.pe_plot(need, title=title, kind='bar', mark_line=None,
                                     mark_point=None, theme='white')


def _plot_twinx_withtarget(st, target, compare_name, yaxis, title, pycharts=False):
    """
    plot_twix_withtarget
    """
    # bar is 'count' or 'mean'
    if yaxis == 'count':
        feas_name = title
    else:
        feas_name = 'rto'

    # plt
    if pycharts is False:
        # get x label
        length = len(compare_name)
        bar_width = 0.9 / length
        x = np.array(range(len(st)))
        tick_label = st['index']

        # plot
        fig = plt.figure(figsize=[20, 10])
        ax1 = fig.add_subplot(111)
        ax = plt.gca()  # 设置x轴刻度的数量
        ax.locator_params("x")
        ax.tick_params(labelsize=20, rotation=90)

        # bar
        for i, name in enumerate(compare_name):
            plt_bar = ax1.bar([c + i * bar_width for c in x], st[f'{name}_{feas_name}'].tolist(), bar_width, alpha=0.5,
                              label=f'{name}')

        # line 
        ax2 = plt.twinx()  # 添加坐标轴
        for name in compare_name:
            plt_target = ax2.plot([c + length * bar_width / 2 for c in x], st[f'{name}_{target}'].tolist(),
                                  alpha=1, linewidth=5, label=f'{name}_{target}')

        ax2.tick_params(labelsize=20)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)

        x_ = x + length * bar_width / 2
        xtick_chunk = 10
        if len(x_) > 50:
            x_ = [int(len(x_) / xtick_chunk) * i for i in range(xtick_chunk)] + [len(x_) - 1]
            tick_label = [tick_label[i] for i in x_]
        plt.title(f'{title}', fontsize=30)
        plt.xticks(x_, tick_label)
        plt.close(fig)

        return fig

    else:
        need = st.set_index('index')[[c for c in st if re.search(feas_name + '|' + target, c)]]
        data_bar = need[[c for c in st if re.search(feas_name, c)]]
        data_bar.rename(columns={f'{name}_{feas_name}': f'{name}' for name in compare_name}, inplace=True)
        data_line = need[[c for c in st if re.search(target, c)]]

        bar = atp.PlotUtils.pe_plot(data_bar, title=title, kind='bar',
                                    mark_line=None, mark_point=None, theme='white')
        if np.nanmax(data_line) > 1:
            param_2nd_y = {'js_code': """function (value) {
                                            return Math.round(value * 100) / 100;
                                        }""",
                           'interval': 0.05 * np.nanmax(data_line)}
        else:
            param_2nd_y = {'js_code': """function (value) {
                                return (Math.floor(value * 10000) / 100) + '%';
                            }""",
                           'interval': 0.05}

        bar = bar.extend_axis(
            yaxis=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(
                    formatter=JsCode(param_2nd_y['js_code'])), interval=param_2nd_y['interval'])
        )
        line = atp.PlotUtils.pe_plot(data_line, title=title, kind='line',
                                     mark_line=None, mark_point=None, theme='white',
                                     yaxis_index=1, z_level=10)
        return bar.overlap(line)


def plot_twinx_feas(compare_dict, feas, target, cut_params, title='feas', yaxis='mean',
                    pycharts=False, save_path=None):
    """
    plot_twinx_feas
    """
    # get state
    st = _get_twix_state(compare_dict, feas,
                         target, cut_params, title)

    # plot
    if target is None:
        if pycharts is False:
            fig = _plot_twinx_withnotarget(st, list(compare_dict.keys()), yaxis, title, pycharts=False)
            if save_path is not None:
                fig.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')
            return st, fig

        else:
            return st, _plot_twinx_withnotarget(st, list(compare_dict.keys()), yaxis, title, pycharts=True)

    else:
        if pycharts is False:
            fig = _plot_twinx_withtarget(st, target, list(compare_dict.keys()), yaxis, title, pycharts=False)
            if save_path is not None:
                fig.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')
            return st, fig

        else:
            return st, _plot_twinx_withtarget(st, target, list(compare_dict.keys()), yaxis, title, pycharts=True)


# noinspection PyUnresolvedReferences
def plot_xgb_trees(model, tree_num, dump=False, node=True, save_path=None):
    """
    params model : booster model
    params tree_num: tree_num
    params dump: whether to dump
    params node: whether to show node
    params save_path: save_path
    """

    if dump is True:
        print(model.get_dump('', with_stats=False, dump_format='text')[tree_num])

    # get all trees
    trees = model.get_dump('', with_stats=False, dump_format='dot')

    # get max_depth
    need = model.trees_to_dataframe()
    # get max_depth
    max_depth = math.ceil(math.log(need['Node'].max(), 2))

    str_ = trees[tree_num]
    # draw tree 
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

    # show node
    if node is True:
        for i in range(2 ** (max_depth + 1) - 1):
            find1 = str_.find(f']\n\n    {i} [ label="leaf=')
            if find1 != -1:
                find2 = find1 + str_[find1:].find('" ]\n')
                str_ = str_[:find1] + str_[find1:find2] + f'\n #node={i}' + str_[find2:]

    # 这样画出来不清晰
    # plt.rcParams['figure.figsize'] = [60, 60]
    # _, ax = plt.subplots(1, 1)
    # s = BytesIO()
    # s.write(Source(str_).pipe(format='png'))
    # s.seek(0)
    # img = image.imread(s)
    # ax.imshow(img)
    # ax.axis('off')
    # plt.savefig(join(save_path, f'tree{tree_num}.png'))

    pic = Source(str_)
    if save_path is not None:
        pic.render(filename=join(save_path, f'tree{tree_num}'), cleanup=True, format='png')
    return pic


def plot_lgb_trees(model, tree_index, precision=3, orientation='vertical', save_path=None):
    pic = lgb.create_tree_digraph(model, tree_index=tree_index, precision=precision, orientation=orientation)
    if save_path is not None:
        pic.render(filename=join(save_path, f'tree{tree_index}'), cleanup=True, format='png')
    return pic


def plot_model_trees(model, tree_num, save_path=None):
    # xgboost model apply func
    if re.search('xgboost', str(model)):
        return plot_xgb_trees(model, tree_num, dump=False, node=True, save_path=save_path)

    # lightgbm model apply func
    elif re.search('lightgbm', str(model)):
        return plot_lgb_trees(model, tree_num, precision=3, orientation='vertical', save_path=save_path)

    else:
        raise ValueError('only support xgb/lgb')


def plot_feas_distribution(data, ref_col, feas,
                           is_stack='stack'):
    """
    is_stack: 'stack' or None
    """
    tmp1 = pd.DataFrame(data.groupby([ref_col])[feas].value_counts()).groupby(level=0).apply(lambda x:
                                                                                             x / float(x.sum())).rename(
        columns={feas: 'rto'}). \
        reset_index()

    tmp2 = pd.DataFrame({ref_col: sorted(list(tmp1[ref_col].unique()) * tmp1[feas].nunique()),
                         feas: list(tmp1[feas].unique()) * tmp1[ref_col].nunique()})

    tmp = tmp1.merge(tmp2, on=[ref_col, feas], how='right').fillna(0).sort_values([feas, ref_col]).reset_index(
        drop=True)
    count = int(tmp.shape[0] / tmp[feas].nunique())
    # xaxis
    b = Bar().add_xaxis([str(c) for c in sorted(data[ref_col].unique())])

    # yaxis
    for i in range(tmp[feas].nunique()):
        b.add_yaxis(str(sorted(tmp[feas].unique())[i]),
                    list(tmp['rto'])[i * count:(i + 1) * count], stack=is_stack, category_gap="50%"). \
            set_series_opts(label_opts=opts.LabelOpts(is_show=False)
                            ).set_global_opts(

            datazoom_opts=opts.DataZoomOpts(),
        )
    return b


def plot_eval_metrics(results, model):
    """
    now only support auc
    """
    metric = 'auc'
    if re.search('catboost', str(model)):
        best_iteration = model.best_iteration_
    else:
        best_iteration = model.best_iteration

    plt.figure(figsize=(20, 10))
    epochs = len(results['train'][metric])
    x_axis = range(0, epochs)

    plt.plot(x_axis, results['train'][metric], label='Train', linewidth=3)
    plt.plot(x_axis, results['test'][metric], label='Test', linewidth=3)

    try:
        plt.plot([best_iteration, best_iteration],
                 [max(results['train'][metric][best_iteration], results['test'][metric][best_iteration]) + 0.01,
                  min(results['train'][metric][0], results['test'][metric][0])],
                 color='k', linewidth=2)
    except IndexError:
        best_iteration = best_iteration - 1
        plt.plot([best_iteration, best_iteration],
                 [max(results['train'][metric][best_iteration], results['test'][metric][best_iteration]) + 0.01,
                  min(results['train'][metric][0], results['test'][metric][0])],
                 color='k', linewidth=2)

    plt.legend(fontsize=20)
    plt.ylabel(f'{metric}', fontsize=25, loc='top')
    plt.tick_params(labelsize=20)
    plt.xlabel('迭代次数', fontsize=25, loc='right')
    plt.title(f'model_{metric}', fontsize=25)
    plt.show()


class OPTPLOT:

    @classmethod
    def plot_convergence(cls, x, y1, y2,
                         xlabel="Number of iterations $n$",
                         ylabel=r"$\min f(x)$ after $n$ iterations",
                         ax=None, name=None, alpha=0.2, yscale=None,
                         color=None, true_minimum=None, **kwargs):
        """Plot one or several convergence traces.

        Parameters
        ----------
        args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
            The result(s) for which to plot the convergence trace.

            - if `OptimizeResult`, then draw the corresponding single trace;
            - if list of `OptimizeResult`, then draw the corresponding convergence
              traces in transparency, along with the average convergence trace;
            - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
              an `OptimizeResult` or a list of `OptimizeResult`.

        ax : `Axes`, optional
            The matplotlib axes on which to draw the plot, or `None` to create
            a new one.

        true_minimum : float, optional
            The true minimum value of the function, if known.

        yscale : None or string, optional
            The scale for the y-axis.

        Returns
        -------
        ax : `Axes`
            The matplotlib axes.
        """
        if ax is None:
            ax = plt.gca()

        ax.set_title("Convergence plot")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.plot(x, y1, c=color, label=name, **kwargs)
        ax.scatter(x, y2, c=color, alpha=alpha)

        if true_minimum:
            ax.axhline(true_minimum, linestyle="--",
                       color="r", lw=1,
                       label="True minimum")

        if true_minimum or name:
            ax.legend(loc="best")
        return ax

    @classmethod
    def plot_hi(cls, data, losses, target_name="loss",
                loss2target_func=None, return_data_only=False):
        if loss2target_func is not None:
            targets = [loss2target_func(loss) for loss in losses]
        else:
            targets = losses
        for config, target in zip(data, targets):
            config[target_name] = target

        if return_data_only:
            return pd.DataFrame(data)
        try:
            import hiplot as hip
        except Exception:
            raise get_import_error("hiplot")

        exp = hip.Experiment.from_iterable(data)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
            'hide': ['from_uid']
        })
        return exp.display()

    @classmethod
    def plot_opt(cls, trials, target_name="accuracy",
                 loss2target_func=lambda x: 1 - x, return_data_only=False):
        losses = []
        for i in range(len(trials)):
            losses.append(trials.trials[i]['result']['loss'])

        params_data = pd.DataFrame(trials.vals).to_dict('record')
        # plot_convergence
        n_calls = len(losses)
        iterations = range(1, n_calls + 1)
        mins = [np.min(losses[:i]) for i in iterations]
        max_mins = max(mins)
        cliped_losses = np.clip(losses, None, max_mins)
        cls.plot_convergence(iterations, mins, cliped_losses)

        return cls.plot_hi(params_data, losses, target_name=target_name, loss2target_func=loss2target_func,
                       return_data_only=return_data_only)
