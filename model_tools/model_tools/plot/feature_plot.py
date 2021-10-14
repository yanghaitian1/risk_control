import re
import sys
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import display
from frozendict import frozendict
from ipywidgets import Layout
from ipywidgets import widgets
from pyecharts import options as opts
from pyecharts.charts import Bar, Line
from pyecharts.commons.utils import JsCode
from sklearn.linear_model import LinearRegression

from ..gentools.advtools import cut_feas
from ..gentools.gentools import df_fillna, rename_df

# Display Chinese
if re.search('^win', sys.platform):
    # Windows (win32)
    font_cn = 'SimHei'
elif re.search('darwin', sys.platform):
    # Mac (darwin) or linux
    font_cn = 'Arial Unicode MS'
else:
    # linux
    font_cn = 'WenQuanYi Zen Hei'

plt.rcParams['font.family'] = font_cn
plt.rcParams['axes.unicode_minus'] = False


class TabOut:
    """
    Use tabout to enhanced visualization
    """

    def __init__(self):
        pass

    @classmethod
    def display_(cls, value):
        """
        display value in tab

        :param value: support dataframe, str, list, plt.fig, pyecharts.fig,etc
        :return: value
        """

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

        :param description:
        :param value:
        :param output:
        :param width:
        :return:
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
        """
        show tab

        :param params: dict, key is the name of tab, value is the value of tab
        :param box_style: support 'success', 'info', 'warning' or 'danger'
        :param column_cnt: the tab cnt in one row
        :return:
        """

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


class PlotUtils:

    @classmethod
    def _get_state(cls, data, feas, target, linreg_fit=False):
        """
        Use groupby to get state

        :param data: dataframe
        :param feas: str: feas name use for {feas: len}
        :param target: None/str/list target name use for {target: 'mean'}
        :param linreg_fit: use LinearRegression to fit target only when target is not None
        :return: state

        Examples:
            x_train['ovd_days_rf_due_sum_p66d_proc'] = pd.cut(x_train['ovd_days_rf_due_sum_p66d'], [-np.inf,0,1,np.inf])
            (1) PlotUtils()._get_state(x_train, feas='ovd_days_rf_due_sum_p66d', target=None, linreg_fit=False)
            (2) PlotUtils()._get_state(x_train, feas='ovd_days_rf_due_sum_p66d_proc', target=['p1_0','p1_1'], linreg_fit=True)
            (3) PlotUtils()._get_state(x_train, feas='ord_cnt_max_in1d_is_holiday_p96d', target='p1_1', linreg_fit=True)
        """

        # update aggfunc
        aggfunc = {feas: len}
        if isinstance(target, str):
            target = [target]
        if target is not None:
            aggfunc.update(dict(zip(target, ['mean'] * len(target))))

        # fillna
        if str(data[feas].dtypes) == 'category':
            if sum(pd.isnull(data[feas])) > 0:
                df_fillna(data, feas, fill_value='NaN', inplace=True)

        state = data.groupby([feas], dropna=False).agg(aggfunc).rename(columns={feas: 'count'})
        state['count_rto'] = state['count'] / data.shape[0]
        if target is not None:
            for target_ in target:
                state[f'{target_}_mean'] = data[target_].mean()
                if linreg_fit:
                    idx = (state.index.isin(['NaN', 'nan', 'na', 'null'])) | (pd.isnull(state.index))
                    state_tmp = state[~idx]
                    state_tmp['index'] = range(state_tmp.shape[0])
                    lin_reg = LinearRegression().fit(state_tmp['index'].values.reshape(-1, 1),
                                                     state_tmp[target_].values.reshape(-1, 1))
                    col_name = str(
                        f'{target_}: coef={round(lin_reg.coef_[0][0], 6)}, intercept={round(lin_reg.intercept_[0], 6)}')
                    state_tmp[col_name] = lin_reg.coef_[0] * list(range(state_tmp.shape[0])) + lin_reg.intercept_[0]
                    state = state.merge(state_tmp[[col_name]], left_index=True, right_index=True, how='left')

        return state

    @classmethod
    def pe_plot(cls, state, cols_name, plot_type='bar', title='feas',
                yaxis_index=0, z_level=0, theme='white', color_list=[], save_path=None):
        """
        Plot by pyecharts, support line, bar, barh

        :param state: datarame
        :param cols_name: column values
        :param plot_type: support line,bar,barh
        :param title: fig title
        :param yaxis_index:
        :param z_level:
        :param theme: theme of the fig, support white,dark,light,etc
        :param color_list: color of each bar or line, for example ['steelblue','darkseagreen','tan','cadetblue']
        :return: fig
        
        Examples:
            (1) PlotUtils().pe_plot(state, cols_name=['count','cnt2'], plot_type='bar', title='feas', theme='white',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue']).render_notebook()
            (2) PlotUtils().pe_plot(state, cols_name=['count','cnt2','cnt3'], plot_type='barh', title='feas', theme='dark',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue']).render_notebook()
            (3) fplt.PlotUtils().pe_plot(state, cols_name=['count','cnt2','cnt3'], plot_type='line', title='feas', theme='white',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue']).render_notebook()
        """

        state.index = state.index.astype(str)
        pe_r_yaxis_opts = frozendict(is_scale=True, min_=None, max_=None)
        pe_legend_opts = frozendict(pos_right='10%', orient='vertical', item_height=3,
                                    item_gap=5, padding=5, legend_icon='circle')

        # Index should be str, or plot wrong
        idx = [str(i) for i in state.index]
        if plot_type == 'line':
            CLS = Line
            params = {'yaxis_index': yaxis_index, 'z_level': z_level}
        elif plot_type in ['bar', 'barh']:
            CLS = Bar
            params = {'yaxis_index': yaxis_index}
        else:
            raise ValueError('Unknown plotting plot_type={} input'.format(plot_type))

            # Plot
        obj = None
        if isinstance(cols_name, str):
            cols_name = [cols_name]

        for i, c_l in enumerate(cols_name):
            if plot_type == 'barh':
                o = Bar(init_opts=opts.InitOpts(theme=theme)) \
                    .set_global_opts(title_opts=opts.TitleOpts(title=title),
                                     datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100,
                                                                      orient="vertical", type_='inside')],
                                     xaxis_opts=opts.AxisOpts(**pe_r_yaxis_opts),
                                     yaxis_opts=opts.AxisOpts(type_='category',
                                                              axislabel_opts=opts.LabelOpts(rotate=0)),
                                     legend_opts=opts.LegendOpts(**pe_legend_opts)
                                     ) \
                    .add_xaxis(idx) \
                    .add_yaxis(str(c_l), state[c_l].tolist(),
                               itemstyle_opts=opts.ItemStyleOpts(
                                   color=color_list[i] if i <= len(color_list) - 1 else None),
                               **params) \
                    .reversal_axis() \
                    .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False, position="right"),
                    linestyle_opts=opts.LineStyleOpts(width=2))

            else:
                o = CLS(init_opts=opts.InitOpts(theme=theme)) \
                    .set_global_opts(title_opts=opts.TitleOpts(title=title),
                                     datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100,
                                                                      type_='inside')],
                                     xaxis_opts=opts.AxisOpts(type_='category',
                                                              axislabel_opts=opts.LabelOpts(rotate=90)),
                                     yaxis_opts=opts.AxisOpts(**pe_r_yaxis_opts),
                                     legend_opts=opts.LegendOpts(**pe_legend_opts)
                                     ) \
                    .add_xaxis(idx) \
                    .add_yaxis(str(c_l), state[c_l].tolist(),
                               itemstyle_opts=opts.ItemStyleOpts(
                                   color=color_list[i] if i <= len(color_list) - 1 else None),
                               **params) \
                    .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False, position="right"),
                    linestyle_opts=opts.LineStyleOpts(width=2))

            if obj is None:
                # First plot
                obj = o
                continue

            obj.overlap(o)

        # save fig
        if save_path is not None:
            obj.render(join(save_path, f'{title}.html'))

        return obj

    @classmethod
    def pe_plot_twinx(cls, state, cols_name, target, title='feas', theme='white',
                      color_bar=[], color_line=[], save_path=None):
        """
        plot twinx bar & line by pyecharts

        :param state: dataframe
        :param cols_name: bar values
        :param target: line values
        :param title: fig title
        :param theme: theme of the fig
        :param color_bar: color of each bar
        :param color_line: color of each line
        :return: fig

        Examples:
            (1) fplt.PlotUtils().pe_plot_twinx(state, cols_name=['count','cnt2'], target=['p1_0'.'p1_1'], title='feas',
                        color_bar=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'],
                        color_line=['red','brown','tomato','violet'],theme='white').render_notebook()
            (2) fplt.PlotUtils().pe_plot_twinx(state, cols_name='count', target='p1_1', title='feas',
                        color_bar=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'],
                        color_line=['red','brown','tomato','violet'],theme='white').render_notebook()
        """

        # plot bar
        bar = cls.pe_plot(state, cols_name, plot_type='bar', title=title, theme=theme, color_list=color_bar)
        if isinstance(target, str):
            target = [target]

        # Extend axis for line plot
        if np.nanmax(state[target]) > 1:
            param_2nd_y = {'js_code': """function (value) {
                                return Math.round(value * 100) / 100;
                            }""",
                           'interval': 0.05 * np.nanmax(state[target])}
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

        line = cls.pe_plot(state, target, plot_type='line', title=title, theme=theme,
                           yaxis_index=1, z_level=10, color_list=color_line)

        bar.overlap(line)

        # save fig
        if save_path is not None:
            bar.render(join(save_path, f'{title}.html'))

        return bar

    @classmethod
    def plt_plot(cls, state, cols_name, plot_type='bar', title='feas',
                 color_list=[], scale_num=20, save_path=None):
        """
        Plot by matplotlib, support line, bar, barh

        :param state: dataframe
        :param cols_name: bar values
        :param plot_type: support bar,barh,line
        :param title: fig title
        :param color_list: color of each bar or line, for example ['steelblue','darkseagreen','tan','cadetblue']
        :param scale_num: scale num in axis, first and last scale will always show
        :return: fig

        Examples:
            (1) PlotUtils().plt_plot(state, cols_name=['count','cnt1','cnt2'], plot_type='bar',  title='feas',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'], scale_num=20)
            (2) fplt.PlotUtils().plt_plot(state, cols_name=['count','cnt1'], plot_type='barh',  title='feas',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'], scale_num=20)
            (3) fplt.PlotUtils().plt_plot(state, cols_name='count', plot_type='line',  title='feas',
                        color_list=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'], scale_num=20)
        """

        state.index = state.index.astype(str)
        fig = plt.figure(figsize=[20, 10], clear=True)

        if isinstance(cols_name, str):
            cols_name = [cols_name]

        length = len(cols_name)
        bar_width = 0.8 / length
        x = np.array(range(len(state)))
        ax1 = fig.add_subplot(111)
        for i, name in enumerate(cols_name):
            if plot_type == 'bar':
                ax1.bar([c + i * bar_width for c in x], state[name].tolist(), bar_width,
                        color=color_list[i] if i <= len(color_list) - 1 else None, alpha=1, label=name)
            elif plot_type == 'barh':
                ax1.barh([c + i * bar_width for c in x], state[name].tolist(), bar_width,
                         color=color_list[i] if i <= len(color_list) - 1 else None, alpha=1, label=name)
            elif plot_type == 'line':
                ax1.plot([c + i * bar_width for c in x], state[name].tolist(),
                         color=color_list[i] if i <= len(color_list) - 1 else None, label=name)
            else:
                raise ValueError('only support bar or barh please correct plot_type')

                # set plt params
        plt.legend(loc='upper right', fontsize=20)
        plt.title(f'{title}', fontsize=30)
        x = [c + min(0.2 * (len(cols_name) - 1), 0.3) for c in x]

        loc = np.quantile(np.arange(0, len(x)), np.arange(0, 1 + 0.000001, 1 / (scale_num + 1))).tolist()
        if state.index[-1] == 'nan':
            loc.append(len(x) - 2)
        if plot_type in ['bar', 'line']:
            plt.xticks(ticks=[x[int(i)] for i in loc], labels=[state.index.tolist()[int(i)] for i in loc])
            plt.tick_params(labelsize=20, rotation=90)
        else:
            plt.yticks(ticks=[x[int(i)] for i in loc], labels=[state.index.tolist()[int(i)] for i in loc])
            plt.tick_params(labelsize=20)

        # save fig
        if save_path is not None:
            plt.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')

        plt.close(fig)

        return fig

    @classmethod
    def plt_plot_twinx(cls, state, cols_name, target, title='feas',
                       color_bar=[], color_line=[], scale_num=20, save_path=None):
        """
        plot twinx bar & line by matplotlib

        :param state: dataframe
        :param cols_name: bar values
        :param target: line values
        :param title: fig title
        :param color_bar: color of each bar
        :param color_line: color of each line
        :param scale_num: scale num in axis, first and last scale will always show
        :return: fig

        Examples:
            (1) fplt.PlotUtils().plt_plot_twinx(state, cols_name=['count','cnt2'], target=['p1_0','p1_1'], title='feas',
                        color_bar=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'],
                        color_line=['red','brown','tomato','violet'],scale_num=20)
            (2) fplt.PlotUtils().plt_plot_twinx(state, cols_name='count', target='p1_0', title='feas',
                        color_bar=['steelblue','darkseagreen','tan','rosybrown','cadetblue','cadetblue'],
                        color_line=['red','brown','tomato','violet'],scale_num=20)
        """

        state.index = state.index.astype(str)
        fig = plt.figure(figsize=[20, 10], clear=True)

        if isinstance(cols_name, str):
            cols_name = [cols_name]

        if isinstance(target, str):
            target = [target]

        length = len(cols_name)
        bar_width = 0.8 / length
        x = np.array(range(len(state)))
        ax1 = fig.add_subplot(111)
        ax = plt.gca()
        ax.locator_params("x")
        ax.tick_params(labelsize=20, rotation=90)

        for i, name in enumerate(cols_name):
            plt_bar = ax1.bar([c + i * bar_width for c in x], state[name].tolist(), bar_width,
                              color=color_bar[i] if i <= len(color_bar) - 1 else None, alpha=0.8, label=name)

        ax2 = plt.twinx()

        for i, name in enumerate(target):
            plt_target = ax2.plot([c + min(0.2 * (len(cols_name) - 1), 0.3) for c in x], state[name].tolist(),
                                  alpha=1, linewidth=2, color=color_line[i] if i <= len(color_line) - 1 else None,
                                  label=name)

        ax2.tick_params(labelsize=20)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=20)

        x = [c + min(0.2 * (len(cols_name) - 1), 0.3) for c in x]
        loc = np.quantile(np.arange(0, len(x)), np.arange(0, 1 + 0.000001, 1 / (scale_num + 1))).tolist()
        if state.index[-1] == 'nan':
            loc.append(len(x) - 2)
        plt.xticks(ticks=[x[int(i)] for i in loc], labels=[state.index.tolist()[int(i)] for i in loc])
        plt.tick_params(labelsize=20, rotation=90)
        plt.title(f'{title}', fontsize=30)

        # save fig
        if save_path is not None:
            plt.savefig(join(save_path, f'{title}.png'), bbox_inches='tight')

        plt.close(fig)

        return fig

    @classmethod
    def plot_bivar(cls, data, feas, target, cut_params=None, title='feas',
                   yaxis='count', pyecharts=False, mark_line=True,
                   draw_lin=False, return_state=False, save_path=None, **kwargs):
        """
        Plot bivar based on feas and target or Plot bar based on feas

        :param data: dataframe
        :param feas: feas in dataframe
        :param target: None/str/list target in dataframe
        :param cut_params: None/int/list/dict method to aggregate feature
        :param title: pic title
        :param yaxis: support count/count_rto, affect the scale of the y-axis
        :param pyecharts: support True/False, use pyecharts or matploblib
        :param mark_line: support True/False, whether draw target mean
        :param draw_lin: support True/False, whether use LinearRegression to fit target
        :param return_state: support True/False, whether return groupby state
        :param save_path: pic save_path
        :param kwargs: thred,max_bins,color_bar,color_line
        :return: groupby state and pic

        Examples:
            (1) PlotUtils().plot_bivar(x_train, feas='ord_cnt_max_in1d_is_holiday_p96d', target=None, cut_params=None, title='feas',
                    yaxis='count', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path='E:/git/yht_tools_new/yht_tools/ytools/')
            (2) PlotUtils().plot_bivar(x_train, feas='normal_repay_amt_rf_due_sum_p7d', target=None,
                    cut_params=10, title='feas2',
                    yaxis='count_rto', pyecharts=True, mark_line=True,
                    draw_lin=False, save_path=None, color_bar=['darkseagreen']).render_notebook()
            (3) PlotUtils().plot_bivar(x_train, feas='ord_cnt_max_in1d_is_holiday_p96d', target='p1_0', cut_params=None, title='feas',
                    yaxis='count', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path=None, color_bar=['steelblue'], color_line=['red','black'])
            (4) state,pic = PlotUtils().plot_bivar(x_train, feas='normal_repay_amt_rf_due_sum_p7d', target='p1_0', cut_params=10, title='feas',
                    yaxis='count_rto', pyecharts=False, mark_line=True,
                    draw_lin=True, color_bar=['steelblue'], color_line=['red','black','cadetblue'],
                    save_path='E:/git/yht_tools_new/yht_tools/ytools/', return_state=True)
            (5) PlotUtils().plot_bivar(x_train, feas='normal_repay_amt_rf_due_sum_p96d', target='p1_0',
                    cut_params={'max_depth': 3, 'min_samples_leaf': 0.05}, title='feas',
                    yaxis='count_rto', pyecharts=False, mark_line=True,
                    draw_lin=True, save_path=None, thred=3, max_bins=3,
                    color_bar=['steelblue'], color_line=['red','cadetblue','black'], return_state=True)
            (6) PlotUtils().plot_bivar(x_train, feas='normal_repay_amt_rf_due_sum_p96d', target=['p1_0','p1_1','p1_3'],
                    cut_params={'max_depth': 3, 'min_samples_leaf': 0.05}, title='feas',
                    yaxis='count_rto', pyecharts=True, mark_line=False,
                    draw_lin=False, save_path=None, thred=3, max_bins=5,
                    color_bar=['steelblue'], color_line=['red','brown','tomato']).render_notebook()
            (7) PlotUtils().plot_bivar(x_train, feas='normal_repay_amt_rf_due_sum_p96d', target=['p1_0','p1_1'],
                    cut_params={'max_depth': 3, 'min_samples_leaf': 0.05}, title='feas',
                    yaxis='count_rto', pyecharts=False, mark_line=True,
                    draw_lin=False, save_path=None, thred=3, max_bins=5,
                    color_bar=['steelblue'], color_line=['red','brown','rosybrown','tan'])
        """

        if target is None:
            df = data[[feas]]
        else:
            if isinstance(target, str):
                target = [target]
            df = data[[feas] + target]

        # get cut bins
        if cut_params is not None:
            if target is None:
                bins = cut_feas(df, feas, target, cut_params,
                                thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            else:
                bins = cut_feas(df, feas, target[0], cut_params,
                                thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            if bins is not None:
                df[feas] = pd.cut(df[feas], bins)

        # get state
        state = cls._get_state(df, feas=feas, target=target, linreg_fit=draw_lin)

        # plot
        if target is None:
            if pyecharts is False:
                pic = cls.plt_plot(state, cols_name=[yaxis], plot_type='bar', title=title,
                                   color_list=kwargs.get('color_bar',
                                                         ['steelblue', 'darkseagreen', 'tan', 'rosybrown',
                                                          'cadetblue']),
                                   scale_num=20, save_path=save_path)
            else:
                pic = cls.pe_plot(state, cols_name=[yaxis], plot_type='bar', title=title, theme='white',
                                  color_list=kwargs.get('color_bar',
                                                        ['steelblue', 'darkseagreen', 'tan', 'rosybrown',
                                                         'cadetblue']), save_path=save_path)

        else:
            if mark_line:
                target = target + [c for c in state.columns if re.search('mean', c)]
            if draw_lin:
                target = target + [c for c in state.columns if re.search('coef', c)]
            state = state[['count', 'count_rto'] + target]

            if pyecharts is False:
                pic = cls.plt_plot_twinx(state, cols_name=[yaxis], target=target, title=title,
                                         color_bar=kwargs.get('color_bar',
                                                              ['steelblue', 'darkseagreen', 'tan',
                                                               'rosybrown', 'cadetblue']),
                                         color_line=kwargs.get('color_line',
                                                               ['red', 'brown', 'tomato', 'violet']),
                                         scale_num=20, save_path=save_path)
            else:
                pic = cls.pe_plot_twinx(state, cols_name=[yaxis], target=target, title=title,
                                        color_bar=kwargs.get('color_bar',
                                                             ['steelblue', 'darkseagreen', 'tan',
                                                              'rosybrown', 'cadetblue']),
                                        color_line=kwargs.get('color_line',
                                                              ['red', 'brown', 'tomato', 'violet']),
                                        theme='white', save_path=save_path)

        if return_state:
            return state, pic

        return pic

    @classmethod
    def plot_bivar_twinx(cls, compare_dict, feas, target, cut_params=None, title='feas',
                         yaxis='count', pyecharts=False, mark_line=True,
                         draw_lin=False, return_state=False, save_path=None, **kwargs):
        """
        Plot bivar twinx based on feas and target or Plot bar twinx based on feas

        :param compare_dict: dict like {'ref':train, 'test':test}
        :param feas: feas in dataframe
        :param target: None/str/list target in dataframe
        :param cut_params: None/int/list/dict method to aggregate feature
        :param title: pic title
        :param yaxis: support count/count_rto, affect the scale of the y-axis
        :param pyecharts: support True/False, use pyecharts or matploblib
        :param mark_line: support True/False, whether draw target mean
        :param draw_lin: support True/False, whether use LinearRegression to fit target
        :param return_state: support True/False, whether return groupby state
        :param save_path: pic save_path
        :param kwargs: thred,max_bins,color_bar,color_line
        :return: groupby state and pic

        Examples:
            compare_dict = {'x_train':x_train, 'x_test':x_test, 'oot':oot}
            (1) state, pic = fplt.PlotUtils().plot_bivar_twinx(compare_dict, feas='ord_cnt_max_in1d_is_holiday_p96d', target=None, cut_params=None,
                             title='feas',yaxis='count', pyecharts=False, mark_line=True,
                             draw_lin=False, return_state=True, save_path=None)
            (2) state, pic = fplt.PlotUtils().plot_bivar_twinx(compare_dict, feas='normal_repay_amt_rf_due_sum_p96d', target=None,
                             cut_params={'n_clusters': 4, 'init': 'k-means++', 'n_init': 10, 'max_iter': 300}, title='feas2',
                             yaxis='count_rto', pyecharts=True, mark_line=True,
                             draw_lin=False, return_state=True, save_path=None,color_bar=['steelblue', 'darkseagreen', 'tan'])
            (3) pic = fplt.PlotUtils().plot_bivar_twinx(compare_dict, feas='normal_repay_amt_rf_due_sum_p96d', target='p1_0',
                            cut_params=10, title='feas2', yaxis='count_rto', pyecharts=True, mark_line=False,
                            draw_lin=False, return_state=False, thred=3, max_bins=5, color_bar=['steelblue', 'darkseagreen', 'tan'],
                            color_line=['red', 'brown', 'tomato'], save_path='E:/git/yht_tools_new/yht_tools/ytools/')
            (4) state, pic = fplt.PlotUtils().plot_bivar_twinx(compare_dict, feas='normal_repay_amt_rf_due_sum_p96d', target='p1_0',
                            cut_params={'max_depth': 3, 'min_samples_leaf': 0.05}, title='feas', yaxis='count_rto', pyecharts=False,
                            mark_line=True, draw_lin=False, return_state=True, save_path=None, thred=3, max_bins=5,
                            color_bar=['steelblue', 'darkseagreen', 'tan'], color_line=['red', 'brown', 'tomato'])
            (5) state, pic = fplt.PlotUtils().plot_bivar_twinx(compare_dict, feas='normal_repay_amt_rf_due_sum_p96d', target=['p1_0','p1_1'],
                            cut_params=10, title='feas2', yaxis='count_rto', pyecharts=True, mark_line=True, draw_lin=True,
                            return_state=True, save_path=None, thred=-1, max_bins=9999, color_bar=['steelblue', 'darkseagreen', 'tan'],
                            color_line=['red', 'brown', 'tomato'])
        """

        # get compare_dict keys and values
        keys_list = list(compare_dict.keys())
        if target is None:
            value_list = [df[[feas]] for df in list(compare_dict.values())]
        else:
            if isinstance(target, str):
                target = [target]
            value_list = [df[[feas] + target] for df in list(compare_dict.values())]

        # get cut bins
        if cut_params is not None:
            if target is None:
                bins = cut_feas(value_list[0], feas, target, cut_params,
                                thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            else:
                bins = cut_feas(value_list[0], feas, target[0], cut_params,
                                thred=kwargs.get('thred', -1), max_bins=kwargs.get('max_bins', 9999))
            if bins is not None:
                for df in value_list:
                    df[feas] = pd.cut(df[feas], bins)

        # get state
        for i, df in enumerate(value_list):
            if i == 0:
                state = cls._get_state(df, feas=feas, target=target, linreg_fit=draw_lin)
                state = rename_df(state, list(state), prefix=keys_list[i] + '_')
            else:
                out = cls._get_state(df, feas=feas, target=target, linreg_fit=draw_lin)
                out = rename_df(out, list(out), prefix=keys_list[i] + '_')
                state = state.merge(out, left_index=True, right_index=True)

        # plot
        if target is None:
            if pyecharts is False:
                pic = cls.plt_plot(state, cols_name=[c for c in state if re.search(f'{yaxis}$', c)],
                                   plot_type='bar', title=title,
                                   color_list=kwargs.get('color_bar',
                                                         ['steelblue', 'darkseagreen', 'tan', 'rosybrown',
                                                          'cadetblue']),
                                   scale_num=20, save_path=save_path)
            else:
                pic = cls.pe_plot(state, cols_name=[c for c in state if re.search(f'{yaxis}$', c)],
                                  plot_type='bar', title=title, theme='white',
                                  color_list=kwargs.get('color_bar',
                                                        ['steelblue', 'darkseagreen', 'tan', 'rosybrown',
                                                         'cadetblue']), save_path=save_path)

        else:
            pattern = '|'.join([str(c) + '$' for c in target])
            target = [c for c in state.columns if re.search(pattern, c)]
            if mark_line:
                target = target + [c for c in state.columns if re.search('mean', c)]
            if draw_lin:
                target = target + [c for c in state.columns if re.search('coef', c)]
            state = state[[c for c in state if re.search('count$|count_rto$', c)] + target]

            if pyecharts is False:
                pic = cls.plt_plot_twinx(state, cols_name=[c for c in state if re.search(f'{yaxis}$', c)],
                                         target=target, title=title,
                                         color_bar=kwargs.get('color_bar',
                                                              ['steelblue', 'darkseagreen', 'tan',
                                                               'rosybrown', 'cadetblue']),
                                         color_line=kwargs.get('color_line',
                                                               ['red', 'brown', 'tomato', 'violet']),
                                         scale_num=20, save_path=save_path)
            else:
                pic = cls.pe_plot_twinx(state, cols_name=[c for c in state if re.search(f'{yaxis}$', c)],
                                        target=target, title=title,
                                        color_bar=kwargs.get('color_bar',
                                                             ['steelblue', 'darkseagreen', 'tan',
                                                              'rosybrown', 'cadetblue']),
                                        color_line=kwargs.get('color_line',
                                                              ['red', 'brown', 'tomato', 'violet']),
                                        theme='white', save_path=save_path)

        if return_state:
            return state, pic

        return pic

    @classmethod
    def plot_distribution(cls, df, by, feas, cut_params=None, pyecharts=True, title='feas'):
        """
        plot feas distribution groupby by

        :param df: df
        :param by: groupby
        :param feas: feas
        :param cut_params: params for cutting feas
        :param pyecharts: True/False
        :param title: fig title
        :return:

        Examples:
            (1) feas = 'ely_gt0_repay_amt_rf_due_mean_p66d_bin_nwoe_bytree'
                plot_distribution(x_train, by='date', feas=feas, pyecharts=True, title=feas).render_notebook()
            (2) feas = 'ely_gt0_repay_amt_rf_due_mean_p66d_bin_nwoe_bytree'
                plot_distribution(x_train, by='date', feas=feas, pyecharts=False, title=feas)
            (3) feas = 'normal_repay_amt_rf_repay_sum_p36d'
                plot_distribution(x_train, by='date', feas=feas, cut_params=3, pyecharts=False, title=feas)
            (4) feas = 'ord_cnt_max_in1d_is_holiday_p186d'
                plot_distribution(x_train, by='date', feas=feas, cut_params=None, pyecharts=True, title=feas).render_notebook()
            (5) feas = 'ord_cnt_max_in1d_is_holiday_p186d'
                plot_distribution(x_train, by='date', feas=feas, cut_params=None, pyecharts=False, title=feas)
        """

        # cut feas
        data = df[[by] + [feas]]
        if cut_params is not None:
            bins = cut_feas(data, feas, target=None, cut_params=cut_params)
            data[feas] = pd.cut(data[feas], bins).astype(str)

        # get distribution
        tmp1 = pd.DataFrame(data.groupby([by])[feas].value_counts(dropna=False)).groupby(level=0).apply(lambda x:
                                                                                                        x / float(
                                                                                                            x.sum())).rename(
            columns={feas: 'rto'}).reset_index()
        tmp2 = pd.DataFrame({by: sorted(list(tmp1[by].unique()) * tmp1[feas].nunique(dropna=False)),
                             feas: list(tmp1[feas].unique()) * tmp1[by].nunique(dropna=False)})
        tmp = tmp1.merge(tmp2, on=[by, feas], how='right').sort_values([feas, by]).reset_index(drop=True)
        tmp['rto'] = tmp['rto'].fillna(0)
        tmp[feas] = tmp[feas].astype(str)
        count = int(tmp.shape[0] / tmp[feas].nunique())

        # draw feas distribution by pyecharts
        if pyecharts:
            pe_legend_opts = frozendict(pos_right='10%', orient='horizontal', item_height=5,
                                        item_gap=2, padding=35, legend_icon='circle')
            # xaxis
            b = Bar().add_xaxis([str(c) for c in sorted(data[by].unique())])
            # yaxis
            for i in range(tmp[feas].nunique()):
                b.add_yaxis(str(sorted(tmp[feas].unique())[i]),
                            list(tmp['rto'])[i * count:(i + 1) * count], stack='stack', category_gap="50%"). \
                    set_series_opts(label_opts=opts.LabelOpts(is_show=False))

            b.set_global_opts(title_opts=opts.TitleOpts(title=f'{title}_distribution'),
                              legend_opts=opts.LegendOpts(**pe_legend_opts),
                              datazoom_opts=opts.DataZoomOpts())

            return b

        # draw feas distribution by matploblib
        else:
            fig = plt.figure(figsize=[20, 10], clear=True)
            bottom = np.array([0] * count)
            for i in range(tmp[feas].nunique()):
                y = list(tmp['rto'])[i * count:(i + 1) * count]
                plt.bar(tmp['date'][:count], y, width=0.8,
                        label=tmp[feas][count * i], bottom=bottom)
                bottom = bottom + np.array(y)

            plt.title(f'{title}_distribution', fontsize=30)
            plt.legend(loc='upper right', fontsize=20)
            x = np.array(range(count))
            loc = np.quantile(np.arange(0, len(x)), np.arange(0, 1 + 0.000001, 1 / (20 + 1))).tolist()
            plt.xticks(ticks=[x[int(i)] for i in loc], labels=[tmp[by][:count].tolist()[int(i)] for i in loc])
            plt.tick_params(labelsize=20, rotation=90)
            plt.close(fig)

            return fig


class OPTPLOT:
    """
    from hyperopt import hp,fmin,tpe,STATUS_OK,Trials,anneal

    1. space:
        hp.choice(label, options)：这可以用于分类参数。它返回一个选项，是一个列表或元组。如: hp.choice("criterion", ["gini","entropy"])
        hp.randint(label, upper)：可以用于整数参数。它返回范围(0,upper)内的一个随机整数。如：hp.randint("max_features",50)
        hp.uniform(label, low, high)：这将在low和high之间统一返回一个值 hp.uniform("max_leaf_nodes",1,10)
        hp.quniform(label, low, high, q)： Returns a value like round(uniform(low, high) / q) * q
        hp.normal(label, mu, sigma)-这将返回一个实际值，该值服从均值为mu和标准差为sigma的正态分布
        hp.qnormal(label, mu, sigma, q)-返回一个类似round(normal(mu, sigma) / q) * q的值
        hp.lognormal(label, mu, sigma)-返回exp(normal(mu, sigma))
        hp.qlognormal(label, mu, sigma, q) -返回一个类似round(exp(normal(mu, sigma)) / q) * q的值

    2. fim:
        随机搜索(hyperopt.rand.suggest)
        模拟退火(hyperopt.anneal.suggest)
        TPE算法（hyperopt.tpe.suggest，算法全称为Tree-structured Parzen Estimator Approach）
    """

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
