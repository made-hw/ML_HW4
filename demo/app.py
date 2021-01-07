import os
import pathlib

import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq

import plotly.graph_objs as go

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("aux_data").resolve()

external_stylesheets = [dbc.themes.BOOTSTRAP, ]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ], )
server = app.server

data_filename = DATA_PATH.joinpath('.csv')

class Binning:
    def __init__(self, gaps, woes, counts, name=None):
        self._name = name
        self._gaps = gaps
        self._woes = woes
        self._counts = counts

factors_list = [
    {'label': 'Возраст организации (ОГРН)', 'value': 'ogrn_age_woe'},
    {'label': 'Возраст организации (юр. адрес)', 'value': 'adr_actual_age_woe'},
    {'label': 'Капитал', 'value': 'ul_capital_sum_woe'},
    {'label': 'Кол-во акционеров', 'value': 'ul_founders_cnt_woe'}]

h_df = pd.DataFrame(
    data=[
        ['Возраст оргранизации (ОГРН)', 0.34],
        ['Возраст организации (юр. адрес)', 0.12],
        ['Капитал', 0.08],
        ['Кол-во акционеров', 0.14]],
    columns=['factor_name', 'gini_score'])

bining = Binning(
    [
        [-100000000000000, 28.5],
        [28.5, 50.5],
        [50.5, 81.5],
        [81.5, 131.5],
        [131.5, 100000000000000]],
    [-0.7440409894342532, -0.42755947853297643, 0.104732426381871, 0.573439441602871, 0.8749172074201356],
    [[1090, 261], [1021, 178], [1164, 119], [1471, 94], [1673, 79]],
    'sdfsg')


def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((1 - 0) * i) + 0
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #96dbfa 0%,
                    #96dbfa {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
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
        )
    trace2 = go.Scatter(
            x=total_bins,
            y=bins_dr,
            yaxis='y2',
            name="Доля дефолтов",
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

forms_el = []
for i in factors_list:
    forms_el.append(
        dbc.FormGroup(
            [
                dbc.Label(i['label']),
                dbc.Col(dbc.Input(type='number', id=i['value'], placeholder="Введите число")),
            ], row=True,
        )
    )

form = dbc.Form(forms_el)

# ######################################
# Dash specific part
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
            html.H5('Модель предсказывет вероятность возврата кредита заёмщиком'
                    'на основе финансовых данных 32 395 компаний (выручка, активы, пассивы и т.д).'),
            html.P('Данное демо является домашней работой по ML (MADE) и является демонстрацией и интерпретацией '
                   'результатов моделирования.'),
        ], ),
        dbc.Col([
            html.A('исходный код данного', href=""),
            html.Br(),
            html.A('код разработки модели', href=""),
        ], width=3),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                dbc.Row([
                    dbc.Col(html.Label("Gini")),
                    dbc.Col(html.B('42%', id="score_text", style={'fontSize': 40, 'color': '#1EAEDB', 'opacity': 0.5})),
                ]),
                id="score",
                className="pretty_container",
            ),
        ], width=4),
        dbc.Col([
            html.Div(
                dbc.Row([
                    dbc.Col(html.Label("Доля дефолтов")),
                    dbc.Col(html.B('6%', id="dr_text", style={'fontSize': 40, 'color': '#1EAEDB', 'opacity': 0.5})),
                ]),
                id="dr",
                className="pretty_container",
            ),
        ], width=4),
        dbc.Col([
            html.Div(className="pretty_container", children=[
                html.Label('Тип клиентов'),
                dcc.Dropdown(id='days',
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

                                data=h_df.to_dict('records'),
                                style_data_conditional=data_bars(h_df, 'gini_score'),
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
                                'интерпритации')
                        ]),
                        dbc.Col([
                            dcc.Dropdown(
                                id='factors',
                                placeholder='Выберите клиентов',
                                options=factors_list,
                                value=factors_list[0]['value'],
                                multi=False),
                            dcc.Graph(id='matplotlib-graph', figure=one_bin_barchart(bining, size=5))
                        ], width=9)
                    ]),
                ]),
                dcc.Tab(label='Скоринг', children=[
                    form, dbc.Alert(f"Вероятность дефолта {7}%", color="info"),
                ]),
            ]),
        ], width=12),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=True)
