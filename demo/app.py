import pathlib
import pickle

import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.graph_objs as go

import pandas as pd
import numpy as np

BASE_PATH = pathlib.Path(__file__).parent.resolve()

external_stylesheets = [dbc.themes.BOOTSTRAP, ]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ], )
server = app.server


# ######################################
# Данные
# ######################################

class Binning:
    def __init__(self, gaps, woes, iv, shares, counts, gini, gaps_counts_shares, gaps_avg=[], hhi=-1, r2=0, name=None):
        self._name = name
        self._gaps = gaps
        self._woes = woes
        self._iv = iv
        self._shares = shares
        self._counts = counts
        self._gini = gini
        self._r2 = r2
        self._hhi = hhi
        self._gaps_counts_shares = gaps_counts_shares
        self._gaps_avg = gaps_avg


with open(BASE_PATH.joinpath('bins_ip.pkl'), 'rb') as f:
    bins_ip = pickle.load(f)

with open(BASE_PATH.joinpath('bins_ul.pkl'), 'rb') as f:
    bins_ul = pickle.load(f)

ip_factors_list = [
    {'label': 'Возраст организации (ОГРН)', 'value': 'ogrn_age', 'gini': 0.345, 'default': 76},
    {'label': 'Возраст организации (юр. адрес)', 'value': 'adr_actual_age_ip', 'gini': 0.125, 'default': 27},
    {'label': 'Капитал', 'value': 'ul_capital_sum', 'gini': 0.080, 'default': 10000},
    {'label': 'Кол-во акционеров', 'value': 'ul_founders_cnt_ip', 'gini': 0.144, 'default': 2}]

ul_factors_list = [
    {'label': 'Чистая прибыль', 'value': 'ar_net_profit', 'gini': 0.304, 'default': 5500000},
    {'label': 'Денежные средства и денежные эквиваленты', 'value': 'ab_cash_and_securities', 'gini': 0.283,
     'default': 172000.0},
    # {'label': 'Долгосрочный обязательства', 'value': 'ab_long_term_liabilities', 'gini': 0.024, 'default': 8000000},
    {'label': 'Срок ведения бизнеса', 'value': 'bus_age', 'gini': 0.194, 'default': 62},
    {'label': 'Срок с момента регистрации юридического адреса', 'value': 'adr_actual_age_ul', 'gini': 0.060,
     'default': 32},
    {'label': 'Количество акционеров', 'value': 'ul_founders_cnt_ul', 'gini': 0.146, 'default': 2},
    {'label': 'Выручка на дебиторскую задолженность', 'value': 'r_1_a', 'gini': 0.191, 'default': 611},
    {'label': 'Себестоимотсь на оборотные активы)', 'value': 'r_2_a', 'gini': 0.067, 'default': 38},
    # {'label': 'Комм. на общие расходы', 'value': 'r_3_a', 'gini': 0.026, 'default': 0},
    {'label': 'Выручка на активы', 'value': 'r_6_a', 'gini': 0.121, 'default': 4},
    {'label': 'Задолженность на активы', 'value': 'r_19_a', 'gini': 0.132, 'default': 4}]

factors_list = [x['value'] for x in ip_factors_list + ul_factors_list]
ip_df = pd.DataFrame(
    data=[[x['label'], x['gini']] for x in ip_factors_list],
    columns=['factor_name', 'gini_score'])
ip_df['gini_score'] = ip_df.gini_score.apply(lambda x: round(x * 100, 2))

ul_df = pd.DataFrame(
    data=[[x['label'], x['gini']] for x in ul_factors_list],
    columns=['factor_name', 'gini_score'])
ul_df['gini_score'] = ul_df.gini_score.apply(lambda x: round(x * 100, 2))

forms_el = []
for i in ip_factors_list:
    forms_el.append(
        html.Div(
            dbc.FormGroup(
                [
                    dbc.Label(i['label']),
                    dbc.Col(dbc.Input(type='number', id=i['value'] + '2', placeholder=i['default'], value=i['default'])),
                ], row=True
            ), id=i['value'] + '1')
    )

for i in ul_factors_list:
    forms_el.append(
        html.Div(
            dbc.FormGroup(
                [
                    dbc.Label(i['label']),
                    dbc.Col(dbc.Input(type='number', id=i['value'] + '2', placeholder=i['default'], value=i['default'])),
                ], row=True
            ), id=i['value'] + '1')
    )
form = dbc.Form(forms_el)


def data_bars(df, column):
    styles = []
    for i in range(1, 101):
        styles.append({
            'if': {
                'filter_query': f'{{{column}}} >= {i} && {{{column}}} < {i + 1}',
                'column_id': column
            },
            'background': (
                f"""
                    linear-gradient(90deg,
                    #96dbfa 0%,
                    #96dbfa {i}%,
                    white {i}%,
                    white 100%)
                """
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles


def one_bin_barchart(bining, size=5):
    total_bins = np.arange(len(bining._woes))
    if len(total_bins) == 0:
        print('No binning for variable', bining._name)
        return

    # высота столбцов - численность
    bins_counts = [b[0] + b[1] for b in bining._counts if len(b) == 2]
    # линия - доля плохих в бакете
    bins_dr = [b[1] / (b[0] + b[1]) for b in bining._counts if len(b) == 2]
    trace1 = go.Bar(
        x=total_bins,
        y=bins_counts,
        yaxis='y1',
        name="Кол-во набл.",
        marker=dict(color='rgba(30, 174, 219, 0.5)'),
    )
    trace2 = go.Scatter(
        x=total_bins,
        y=bins_dr,
        yaxis='y2',
        name="Доля дефолтов",
        marker=dict(color='rgba(30, 174, 219, 1)'),
    )

    data = [trace1, trace2]
    layout = go.Layout(
        yaxis=dict(title='Число наблюдений в бакете'),
        yaxis2=dict(title='Уровень дефолтов', overlaying='y', side='right'),
        plot_bgcolor='white',
    )
    fig = go.Figure(data=data, layout=layout)
    x_labels = ['<=' + str(gap[1]) for gap in bining._gaps]
    x_labels[-1] = '< +inf'
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=total_bins,
            ticktext=x_labels,
        )
    )

    return fig


# ######################################
# Структура
# ######################################

# The app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Предсказание дефолта'),
        ], ),
        dbc.Col([
            html.A(html.Img(src=app.get_asset_url("main_logo.svg")), href=""),
            html.Div([html.B('Разработчики: '), html.P('Иванов А.В. и Горяев А.Ф.')], ),
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            html.H5('Модель предсказывет вероятность возврата кредита заёмщиком '
                    'на основе финансовых данных 32 395 компаний (выручка, активы, пассивы и т.д).'),
            html.P('Данное демо является домашней работой по ML (MADE) и является демонстрацией и интерпретацией '
                   'результатов моделирования.'),
        ], ),
        dbc.Col([
            html.A('исходный код данного демо',
                   href="https://github.com/made-hw/ML_HW4/tree/master/demo"),
            html.Br(),
            html.A('код разработки модели', href="https://github.com/made-hw/ML_HW4/blob/master/Solution_and_presenattion.ipynb"),
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                dbc.Row([
                    dbc.Col(html.Label("Gini")),
                    dbc.Col(html.B('40%', id="score_text", style={'fontSize': 40, 'color': '#1EAEDB', 'opacity': 0.5})),
                ]),
                id="score",
                className="pretty_container",
            ),
        ], width=4),
        dbc.Col([
            html.Div(
                dbc.Row([
                    dbc.Col(html.Label("Доля дефолтов")),
                    dbc.Col(html.B('10.2%', id="dr_text", style={'fontSize': 40, 'color': '#1EAEDB', 'opacity': 0.5})),
                ]),
                id="dr",
                className="pretty_container",
            ),
        ], width=4),
        dbc.Col([
            html.Div(className="pretty_container", children=[
                html.Label('Тип клиентов'),
                dcc.Dropdown(id='cl_type',
                             placeholder='Выберите клиентов',
                             options=[{'label': 'Без фин. отчётн.', 'value': 0},
                                      {'label': 'С фин. отчётностью', 'value': 1}, ],
                             value=0,
                             multi=False),
            ]),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Tabs(id='tab', children=[
                dcc.Tab(label='Анализ факторов', children=[
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H2('Однофакторный анализ'),
                            html.Br(),
                            html.P(
                                'Влияние каждого фактора на целевое событие для того, чтобы выявить самые важные факторы.')
                        ]),
                        dbc.Col([
                            dash_table.DataTable(
                                id='table',
                                columns=[
                                    {'name': 'Фактор', 'id': 'factor_name'},
                                    {'name': 'Gini', 'id': 'gini_score'},
                                ],
                                data=ip_df.to_dict('records'),
                                style_data_conditional=data_bars(ip_df, 'gini_score'),
                                style_as_list_view=True,
                            )
                        ])
                    ]),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.H2('Зависимость целевого события'),
                            html.Br(),
                            html.P(
                                'Над факторами было совершенно монотонное WOE-преобразование, в том числе для легкой '
                                'интерпретации')
                        ]),
                        dbc.Col([
                            dcc.Dropdown(
                                id='factors',
                                placeholder='Выберите фактор',
                                options=[{'label': x['label'], 'value': x['value']} for x in ip_factors_list],
                                value=bins_ip[0]._name,
                                multi=False),
                            dcc.Graph(id='matplotlib-graph', figure=one_bin_barchart(bins_ip[0], size=5))
                        ], width=9)
                    ]),
                ]),
                dcc.Tab(label='Скоринг', children=[
                    html.Div(form, id='score_tab'), dbc.Alert(f"Вероятность дефолта {7}%", color="info", id='alert'),
                ]),
            ]),
        ], width=12),
    ]),
])


# ######################################
#  Обновления
# ######################################

@app.callback(
    [
        Output("score_text", "children"),
        Output("dr_text", "children"),
        Output("table", "data"),
        Output("table", "style_data_conditional"),
        Output("factors", "options"),
        Output("factors", "value"),
        *[Output(x + '1', 'style') for x in factors_list],
        *[Output(x + '2', 'value') for x in factors_list],
    ],
    Input("cl_type", "value"),
)
def update_text(data):
    if data == 0:
        return ('40.0%', '10.2%', ip_df.to_dict('records'), data_bars(ip_df, 'gini_score'), [
            {'label': x['label'], 'value': x['value']} for x in ip_factors_list], ip_factors_list[0]['value'], \
                *[{'display': 'block'} if x in [x['value'] for x in ip_factors_list] else {'display': 'none'} for x in
                  factors_list], *[0 if x not in [x['value'] for x in ip_factors_list] else [y['default'] for y in ip_factors_list if y['value'] == x][0] for x in
                  factors_list])
    else:
        return ('42.1%', '4.7%', ul_df.to_dict('records'), data_bars(ul_df, 'gini_score'), [
            {'label': x['label'], 'value': x['value']} for x in ul_factors_list], ul_factors_list[0]['value'], \
                *[{'display': 'block'} if x in [x['value'] for x in ul_factors_list] else {'display': 'none'} for x in
                  factors_list], *[0 if x not in [x['value'] for x in ul_factors_list] else [y['default'] for y in ul_factors_list if y['value'] == x][0] for x in
                  factors_list])


@app.callback(
    Output("matplotlib-graph", "figure"),
    [Input("cl_type", "value"), Input("factors", "value"), ]
)
def update_graph(data1, data2, ):
    if data1 == 0:
        return one_bin_barchart([x for x in bins_ip if x._name == data2 or x._name + '_ip' == data2][0], size=5)
    else:
        return one_bin_barchart([x for x in bins_ul if x._name == data2 or x._name + '_ul' == data2][0], size=5)


@app.callback(
    Output("alert", "children"),
    [
        Input("cl_type", "value"), *[Input(x + '2', 'value') for x in factors_list]
    ]
)
def update_calc(data1, *data2):
    if data1 == 0:
        data2 = [0 if x is None else x for x in data2][:4]
        take_bin = lambda name: [x for x in bins_ip if x._name == name or x._name + '_ip' == name][0]
        take_woe = lambda v, name: take_bin(name)._woes[
            [(i, x) for (i, x) in enumerate(take_bin(name)._gaps) if x[0] <= v < x[1]][0][0]]
        lr = np.sum([x[0] * x[1] for x in zip([take_woe(x[1], x[0]['value']) for x in zip(ip_factors_list, data2)],
                                              [-0.94890533, -0.65224689, -0.86978949, -0.96749802])])
        lr += -2.18364137
        return f'Вероятность дефолта {round(np.exp(lr) / (1 + np.exp(lr)) * 100, 2)} %'
    else:
        take_bin = lambda name: [x for x in bins_ul if x._name == name or x._name + '_ul' == name][0]
        take_woe = lambda v, name: take_bin(name)._woes[
            [(i, x) for (i, x) in enumerate(take_bin(name)._gaps) if x[0] <= v < x[1]][0][0]]
        lr = np.sum([x[0] * x[1] for x in zip([take_woe(x[1], x[0]['value']) for x in zip(ul_factors_list, data2)],
                                              [-0.63756363, -0.47038324, -0.58163822, -1.11727878, -0.47083305,
                                               -0.31323299, 0.22422398, -0.27440618, -0.55363363])])
        lr += -3.0058736
        return f'Вероятность дефолта {round(np.exp(lr) / (1 + np.exp(lr)) * 100, 2)} %'


if __name__ == '__main__':
    app.run_server(debug=False)
