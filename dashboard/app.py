import base64
import io
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, ctx, dash_table, dcc, html

# ── Paths & constants ─────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "disaster_model.pkl"
DATA_PATH = ROOT / "ml" / "data" / "Dados ESP32 GS.xlsx"

CLASSES = {
    0: ("Sem Desastre",           "#2ecc71"),
    1: ("Alerta de Incêndio",     "#f39c12"),
    2: ("Incêndio",               "#e74c3c"),
    3: ("Alerta de Deslizamento", "#f39c12"),
    4: ("Deslizamento",           "#e74c3c"),
    5: ("Alerta de Enchente",     "#f39c12"),
    6: ("Enchente",               "#e74c3c"),
}

FEATURES = [
    "Temperatura (ºC)",
    "Umidade (%)",
    "Luminosidade (LUX)",
    "Nivel da agua",
    "Vibração do solo",
    "Tempo_Minutos",
    "Hora_sin",
    "Hora_cos",
]

SENSOR_COLS = [
    "Temperatura (ºC)",
    "Umidade (%)",
    "Luminosidade (LUX)",
    "Nivel da agua",
    "Vibração do solo",
]

SENSOR_COLORS = {
    "Temperatura (ºC)":   "#e74c3c",
    "Umidade (%)":        "#4fc3f7",
    "Luminosidade (LUX)": "#f39c12",
    "Nivel da agua":      "#3498db",
    "Vibração do solo":   "#9b59b6",
}

# ── Model & demo data (loaded once at startup) ────────────────────────────────

modelo = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _engenharia_temporal(df: pd.DataFrame) -> pd.DataFrame:
    if "Data" not in df.columns:
        return df
    df = df.copy()
    df["Dia"] = df["Data"].str.extract(r"Dia (\d+)").astype(int)
    df["Hora"] = df["Data"].str.extract(r"(\d+):(\d+)")[0].astype(int)
    df["Minuto"] = df["Data"].str.extract(r"(\d+):(\d+)")[1].astype(int)
    df["Tempo_Minutos"] = (df["Dia"] - 1) * 24 * 60 + df["Hora"] * 60 + df["Minuto"]
    hora_dia_min = df["Hora"] * 60 + df["Minuto"]
    df["Hora_sin"] = np.sin(2 * np.pi * hora_dia_min / 1440)
    df["Hora_cos"] = np.cos(2 * np.pi * hora_dia_min / 1440)
    return df


def _load_demo() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_excel(DATA_PATH)
    return _engenharia_temporal(df)


df_demo = _load_demo()

# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_csv(contents: str) -> pd.DataFrame:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))
    df.columns = df.columns.str.strip()
    return df


def _dark_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _kpi_body(label: str, value: str, sub: str = "") -> dbc.CardBody:
    return dbc.CardBody([
        html.P(label, className="text-secondary mb-1",
               style={"fontSize": "0.68rem", "textTransform": "uppercase",
                      "letterSpacing": "0.12em", "fontWeight": "600"}),
        html.H3(value, className="mb-0",
                style={"fontWeight": "800", "color": "#e6edf3", "fontSize": "1.6rem"}),
        html.Small(sub, className="text-secondary") if sub else None,
    ])

# ── App init ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Disaster Monitor",
    update_title=None,
)
app.config.suppress_callback_exceptions = True

# ── Layout ────────────────────────────────────────────────────────────────────

_UPLOAD_STYLE = {
    "width": "100%",
    "padding": "22px",
    "textAlign": "center",
    "borderRadius": "8px",
    "border": "2px dashed #30363d",
    "backgroundColor": "#161b22",
    "cursor": "pointer",
    "color": "#8b949e",
}

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#0d1117", "minHeight": "100vh", "padding": "0"},
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "borderBottom": "1px solid #21262d",
                "background": "linear-gradient(180deg, #161b22 0%, #0d1117 100%)",
                "padding": "16px 24px",
            },
            children=dbc.Container(fluid=True, children=dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span(className="pulse-dot me-2"),
                        html.Span("DISASTER MONITOR",
                                  style={
                                      "color": "#e6edf3",
                                      "fontWeight": "800",
                                      "fontSize": "1.1rem",
                                      "letterSpacing": "0.08em",
                                  }),
                    ], className="d-flex align-items-center mb-1"),
                    html.P(
                        "ESP32 IoT + Machine Learning — Monitoramento Ambiental em Tempo Real",
                        className="mb-0 text-secondary",
                        style={"fontSize": "0.78rem"},
                    ),
                ], width=True),
                dbc.Col(
                    html.Div([
                        html.Span("● SISTEMA ONLINE",
                                  style={
                                      "color": "#2ecc71",
                                      "fontSize": "0.7rem",
                                      "fontWeight": "700",
                                      "letterSpacing": "0.15em",
                                  }),
                    ], className="text-end d-flex align-items-center justify-content-end h-100"),
                    width="auto",
                ),
            ], className="align-items-center")),
        ),

        # ── Tabs ─────────────────────────────────────────────────────────────
        dbc.Container(fluid=True, style={"padding": "0 24px"}, children=[
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-monitor",
                className="mt-3",
                children=[
                    dbc.Tab(label="📡  Monitor",  tab_id="tab-monitor"),
                    dbc.Tab(label="🔮  Previsão", tab_id="tab-previsao"),
                    dbc.Tab(label="📊  Insights", tab_id="tab-insights"),
                ],
            ),
            html.Div(id="tab-content", style={"paddingTop": "24px", "paddingBottom": "40px"}),
        ]),
    ],
)

# ── Tab builders ─────────────────────────────────────────────────────────────


def _build_monitor():
    initial_data = df_demo.to_json(orient="split", date_format="iso") if df_demo is not None else None
    if df_demo is not None:
        initial_alert = dbc.Alert(
            "Exibindo dados de demonstração. Faça upload de um CSV do ESP32 (modo admin) para usar dados reais.",
            color="info", dismissable=True, className="mb-0",
        )
    elif initial_data is None:
        initial_alert = dbc.Alert(
            "Dados não encontrados. Execute o notebook ML para gerar os arquivos necessários.",
            color="warning", className="mb-0",
        )
    else:
        initial_alert = None

    return html.Div([
        dcc.Store(id="store-sensor-data", data=initial_data),

        # Upload
        dbc.Row(dbc.Col(
            dcc.Upload(
                id="upload-sensor",
                className="upload-zone",
                style=_UPLOAD_STYLE,
                children=html.Div([
                    html.Span("⬆  ", style={"fontSize": "1.2rem"}),
                    "Arraste ou ",
                    html.A("selecione um CSV", style={"color": "#e74c3c", "textDecoration": "none"}),
                    html.Br(),
                    html.Small("Formato ESP32 modo admin (Data, ID, sensores, Saída ML)",
                               className="text-secondary"),
                ]),
            ), md=12), className="mb-3"),

        # Alert bar — pre-populated, updated on upload
        html.Div(id="sensor-alert-bar", children=initial_alert, className="mb-3"),

        # KPI cards
        dbc.Row([
            dbc.Col(dbc.Card(id="kpi-total", className="kpi-card h-100"), md=6, lg=3),
            dbc.Col(dbc.Card(id="kpi-risk",  className="kpi-card h-100"), md=6, lg=3),
            dbc.Col(dbc.Card(id="kpi-temp",  className="kpi-card h-100"), md=6, lg=3),
            dbc.Col(dbc.Card(id="kpi-water", className="kpi-card h-100"), md=6, lg=3),
        ], className="mb-4 g-3"),

        # Charts
        html.Div(id="sensor-charts"),
    ])


def _build_previsao():
    if modelo is None:
        return dbc.Alert(
            "Modelo não encontrado. Execute o notebook ML para gerar ml/models/disaster_model.pkl.",
            color="danger",
        )
    return html.Div([
        dcc.Store(id="store-pred-data"),

        dbc.Row(dbc.Col([
            html.P("MODO DE ENTRADA", className="text-secondary mb-2",
                   style={"fontSize": "0.68rem", "letterSpacing": "0.15em", "fontWeight": "600"}),
            dbc.RadioItems(
                id="pred-mode",
                options=[
                    {"label": "  Entrada manual", "value": "manual"},
                    {"label": "  Upload de CSV",  "value": "csv"},
                ],
                value="manual",
                inline=True,
                className="mb-4",
                inputStyle={"marginRight": "6px"},
                labelStyle={"marginRight": "24px", "color": "#8b949e"},
            ),
        ], md=12), className="mb-1"),

        html.Div(id="pred-input-area", children=_manual_sliders()),
        html.Div(id="pred-result", className="mt-4"),
    ])


def _manual_sliders():
    _sl = {"marginBottom": "28px"}
    return dbc.Row([
        dbc.Col([
            html.P("Temperatura (°C)", className="sensor-label"),
            dcc.Slider(id="sl-temp", min=-10, max=90, step=0.5, value=25.0,
                       marks={-10: "-10", 0: "0", 30: "30", 60: "60", 90: "90"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style=_sl),

            html.P("Umidade (%)", className="sensor-label"),
            dcc.Slider(id="sl-umid", min=0, max=100, step=1, value=50,
                       marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100%"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style=_sl),

            html.P("Luminosidade (LUX)", className="sensor-label"),
            dcc.Slider(id="sl-lux", min=0, max=25000, step=100, value=1200,
                       marks={0: "0", 5000: "5k", 15000: "15k", 25000: "25k"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], md=6),

        dbc.Col([
            html.P("Nível da Água (cm)", className="sensor-label"),
            dcc.Slider(id="sl-nivel", min=0, max=400, step=1, value=200,
                       marks={0: "0", 100: "100", 200: "200", 300: "300", 400: "400"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style=_sl),

            html.P("Vibração do Solo (ADC)", className="sensor-label"),
            dcc.Slider(id="sl-vibr", min=2000, max=4095, step=1, value=2500,
                       marks={2000: "2000", 3000: "3000", 4095: "4095"},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(style=_sl),

            dbc.Row([
                dbc.Col([
                    html.P("Hora do dia", className="sensor-label"),
                    dcc.Slider(id="sl-hora", min=0, max=23, step=1, value=12,
                               marks={0: "0h", 6: "6h", 12: "12h", 18: "18h", 23: "23h"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], md=8),
                dbc.Col([
                    html.P("Minuto", className="sensor-label"),
                    dcc.Dropdown(
                        id="dd-minuto",
                        options=[{"label": "00 min", "value": 0},
                                 {"label": "30 min", "value": 30}],
                        value=0,
                        clearable=False,
                    ),
                ], md=4),
            ]),
        ], md=6),

        dbc.Col(
            dbc.Button(
                "🔮  Prever Desastre", id="btn-prever",
                color="danger", size="lg", className="w-100 mt-3",
                style={"letterSpacing": "0.06em", "fontWeight": "700"},
            ),
            md=12,
        ),
    ], className="g-3")


def _csv_upload_pred():
    return html.Div([
        dcc.Upload(
            id="upload-pred",
            className="upload-zone",
            style=_UPLOAD_STYLE,
            children=html.Div([
                html.Span("⬆  ", style={"fontSize": "1.2rem"}),
                "Arraste ou ",
                html.A("selecione CSV de sensores", style={"color": "#e74c3c", "textDecoration": "none"}),
                html.Br(),
                html.Small("Colunas: Temperatura, Umidade, Luminosidade, Nivel da agua, Vibração do solo",
                           className="text-secondary"),
            ]),
        )
    ])


def _build_insights():
    if modelo is None:
        return dbc.Alert("Modelo não encontrado. Execute o notebook ML primeiro.", color="danger")

    # Feature importance
    df_imp = (
        pd.DataFrame({"Feature": FEATURES, "Importância": modelo.feature_importances_})
        .sort_values("Importância")
    )
    fig_imp = px.bar(
        df_imp, x="Importância", y="Feature", orientation="h",
        title="Importância das Features",
        color="Importância",
        color_continuous_scale=[[0, "#21262d"], [1, "#e74c3c"]],
        labels={"Importância": "Importância Relativa"},
    )
    _dark_fig(fig_imp)
    fig_imp.update_layout(coloraxis_showscale=False, yaxis_title="")

    charts: list = [
        dbc.Row(dbc.Col(
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_imp, config={"displayModeBar": False})
            ), className="kpi-card"),
        ), className="mb-4"),
    ]

    # Class distribution
    if df_demo is not None:
        col = next(
            (c for c in ("Tipo_Desastre", "Saída ML") if c in df_demo.columns),
            None,
        )
        if col:
            nome_map = {k: v[0] for k, v in CLASSES.items()}
            cor_map = {v[0]: v[1] for v in CLASSES.values()}
            contagem = df_demo[col].map(nome_map).value_counts()
            fig_dist = px.pie(
                values=contagem.values,
                names=contagem.index,
                title="Distribuição de Classes no Dataset de Treino",
                color=contagem.index,
                color_discrete_map=cor_map,
            )
            _dark_fig(fig_dist)
            charts.append(dbc.Row(dbc.Col(
                dbc.Card(dbc.CardBody(
                    dcc.Graph(figure=fig_dist, config={"displayModeBar": False})
                ), className="kpi-card"),
            ), className="mb-4"))

    # Model parameters table
    params = [
        {"Parâmetro": "Algoritmo",    "Valor": "Random Forest Classifier"},
        {"Parâmetro": "Estimadores",  "Valor": str(modelo.n_estimators)},
        {"Parâmetro": "Class Weight", "Valor": str(modelo.class_weight)},
        {"Parâmetro": "Classes",      "Valor": str(len(modelo.classes_))},
        {"Parâmetro": "Features",     "Valor": str(len(FEATURES))},
    ]
    _TABLE_STYLE = dict(
        style_as_list_view=True,
        style_header={
            "backgroundColor": "#161b22", "color": "#e6edf3",
            "fontWeight": "700", "border": "1px solid #30363d",
            "fontSize": "0.75rem", "textTransform": "uppercase", "letterSpacing": "0.08em",
        },
        style_data={"backgroundColor": "#0d1117", "color": "#c9d1d9", "border": "1px solid #21262d"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#0d1117"}],
        style_cell={"padding": "10px 16px", "fontFamily": "monospace", "fontSize": "0.83rem"},
    )
    table = dash_table.DataTable(
        data=params,
        columns=[{"name": "Parâmetro", "id": "Parâmetro"}, {"name": "Valor", "id": "Valor"}],
        **_TABLE_STYLE,
    )
    charts.append(dbc.Row(dbc.Col([
        html.P("PARÂMETROS DO MODELO", className="text-secondary mb-3",
               style={"fontSize": "0.68rem", "letterSpacing": "0.15em", "fontWeight": "600"}),
        dbc.Card(dbc.CardBody(table), className="kpi-card"),
    ], md=12), className="mb-4"))

    return html.Div(charts)


# ── Callbacks ─────────────────────────────────────────────────────────────────


@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(active_tab: str):
    if active_tab == "tab-monitor":
        return _build_monitor()
    if active_tab == "tab-previsao":
        return _build_previsao()
    return _build_insights()


# ── Tab 1: parse upload or fall back to demo ──────────────────────────────────

@app.callback(
    Output("store-sensor-data", "data"),
    Output("sensor-alert-bar", "children"),
    Input("upload-sensor", "contents"),
    Input("upload-sensor", "filename"),
    prevent_initial_call=True,
)
def store_sensor_data(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update
    df = _parse_csv(contents)
    alert = dbc.Alert(
        [html.Strong("Arquivo carregado: "), filename],
        color="success", dismissable=True, duration=4000, className="mb-0",
    )
    return df.to_json(orient="split", date_format="iso"), alert


# ── Tab 1: KPI cards ──────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-total",  "children"),
    Output("kpi-total",  "className"),
    Output("kpi-risk",   "children"),
    Output("kpi-risk",   "className"),
    Output("kpi-temp",   "children"),
    Output("kpi-temp",   "className"),
    Output("kpi-water",  "children"),
    Output("kpi-water",  "className"),
    Input("store-sensor-data", "data"),
)
def update_kpis(data):
    base = "kpi-card h-100"
    placeholder = _kpi_body("—", "N/A")

    if data is None:
        return (placeholder, base) * 4

    df = pd.read_json(io.StringIO(data), orient="split")
    total = len(df)

    # Risk
    if "Saída ML" in df.columns:
        n_risk = int((df["Saída ML"] > 0).sum())
        pct = n_risk / total * 100 if total > 0 else 0
        risk_cls = (
            f"{base} glow-red" if pct >= 20 else
            f"{base} glow-yellow" if pct > 0 else
            f"{base} glow-green"
        )
        risk_body = _kpi_body("Leituras com Risco", f"{n_risk:,}", f"{pct:.1f}% do total")
    else:
        risk_cls = base
        risk_body = _kpi_body("Leituras com Risco", "N/D")

    temp_body = (
        _kpi_body("Temp. Máxima", f"{df['Temperatura (ºC)'].max():.1f} °C")
        if "Temperatura (ºC)" in df.columns
        else _kpi_body("Temp. Máxima", "N/D")
    )
    water_body = (
        _kpi_body("Nível Máx. Água", f"{df['Nivel da agua'].max():.0f} cm")
        if "Nivel da agua" in df.columns
        else _kpi_body("Nível Máx. Água", "N/D")
    )

    return (
        _kpi_body("Total de Leituras", f"{total:,}"), base,
        risk_body, risk_cls,
        temp_body, base,
        water_body, base,
    )


# ── Tab 1: sensor line charts ─────────────────────────────────────────────────

@app.callback(
    Output("sensor-charts", "children"),
    Input("store-sensor-data", "data"),
)
def update_sensor_charts(data):
    if data is None:
        return []
    df = pd.read_json(io.StringIO(data), orient="split")
    eixo_x = "Tempo_Minutos" if "Tempo_Minutos" in df.columns else df.columns[0]
    charts = []
    for col in [c for c in SENSOR_COLS if c in df.columns]:
        color = SENSOR_COLORS.get(col, "#4fc3f7")
        fig = px.line(df, x=eixo_x, y=col, labels={eixo_x: "Tempo (min)", col: col})
        _dark_fig(fig)
        fig.update_traces(line=dict(color=color, width=1.5))
        fig.update_layout(title=dict(text=col, font=dict(size=13, color="#8b949e")))
        charts.append(dbc.Card(
            dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})),
            className="kpi-card mb-3",
        ))
    return charts


# ── Tab 2: mode toggle ────────────────────────────────────────────────────────

@app.callback(
    Output("pred-input-area", "children"),
    Input("pred-mode", "value"),
)
def toggle_pred_mode(mode: str):
    return _manual_sliders() if mode == "manual" else _csv_upload_pred()


# ── Tab 2: CSV prediction store ───────────────────────────────────────────────

@app.callback(
    Output("store-pred-data", "data"),
    Input("upload-pred", "contents"),
    prevent_initial_call=True,
)
def parse_pred_csv(contents):
    if contents is None:
        return None
    df = _parse_csv(contents)
    df = _engenharia_temporal(df)
    return df.to_json(orient="split", date_format="iso")


# ── Tab 2: unified prediction output ─────────────────────────────────────────

@app.callback(
    Output("pred-result", "children"),
    Input("store-pred-data", "data"),
    Input("btn-prever", "n_clicks"),
    State("sl-temp",   "value"),
    State("sl-umid",   "value"),
    State("sl-lux",    "value"),
    State("sl-nivel",  "value"),
    State("sl-vibr",   "value"),
    State("sl-hora",   "value"),
    State("dd-minuto", "value"),
    prevent_initial_call=True,
)
def handle_prediction(store_data, _n_clicks, temp, umid, lux, nivel, vibr, hora, minuto):
    triggered = ctx.triggered_id

    # ── CSV batch prediction ──────────────────────────────────────────────────
    if triggered == "store-pred-data":
        if store_data is None:
            return None
        df = pd.read_json(io.StringIO(store_data), orient="split")
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return dbc.Alert(
                [html.Strong("Colunas ausentes: "), ", ".join(missing)],
                color="warning",
            )
        preds = modelo.predict(df[FEATURES])
        nome_map = {k: v[0] for k, v in CLASSES.items()}
        cor_map = {v[0]: v[1] for v in CLASSES.values()}
        df_out = df[[c for c in SENSOR_COLS if c in df.columns]].copy()
        df_out["Previsão"] = [CLASSES[int(p)][0] for p in preds]
        contagem = pd.Series(preds).map(nome_map).value_counts()
        fig_pizza = px.pie(
            values=contagem.values,
            names=contagem.index,
            title="Distribuição das Previsões",
            color=contagem.index,
            color_discrete_map=cor_map,
        )
        _dark_fig(fig_pizza)
        _TABLE = dict(
            style_header={"backgroundColor": "#161b22", "color": "#e6edf3", "fontWeight": "700"},
            style_data={"backgroundColor": "#0d1117", "color": "#c9d1d9"},
            style_cell={"border": "1px solid #21262d", "padding": "8px 12px", "fontSize": "0.82rem"},
        )
        table = dash_table.DataTable(
            data=df_out.head(100).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_out.columns],
            page_size=10,
            **_TABLE,
        )
        return html.Div([
            html.P(f"{len(preds)} previsões geradas", className="text-secondary mb-3",
                   style={"fontSize": "0.78rem"}),
            dbc.Card(dbc.CardBody(table), className="kpi-card mb-4"),
            dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_pizza, config={"displayModeBar": False})
            ), className="kpi-card"),
        ])

    # ── Manual single prediction ──────────────────────────────────────────────
    if None in (temp, umid, lux, nivel, vibr, hora, minuto):
        return None

    hora_dia_min = hora * 60 + int(minuto)
    hora_sin = float(np.sin(2 * np.pi * hora_dia_min / 1440))
    hora_cos = float(np.cos(2 * np.pi * hora_dia_min / 1440))

    X = pd.DataFrame(
        [[temp, umid, lux, nivel, vibr, hora_dia_min, hora_sin, hora_cos]],
        columns=FEATURES,
    )
    pred = int(modelo.predict(X)[0])
    probas = modelo.predict_proba(X)[0]
    nome, cor = CLASSES[pred]

    df_probas = (
        pd.DataFrame({
            "Classe": [CLASSES[c][0] for c in modelo.classes_],
            "Probabilidade": probas,
        })
        .sort_values("Probabilidade", ascending=True)
    )
    fig_bar = px.bar(
        df_probas, x="Probabilidade", y="Classe", orientation="h",
        range_x=[0, 1],
        title="Probabilidade por Classe",
        color="Probabilidade",
        color_continuous_scale=[[0, "#21262d"], [0.5, "#f39c12"], [1, "#e74c3c"]],
    )
    _dark_fig(fig_bar)
    fig_bar.update_layout(coloraxis_showscale=False, yaxis_title="")

    return html.Div([
        dbc.Row(dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.P("RESULTADO DA PREVISÃO", className="text-secondary mb-2",
                           style={"fontSize": "0.68rem", "letterSpacing": "0.15em",
                                  "fontWeight": "600"}),
                    html.H2(nome,
                            style={"color": cor, "fontWeight": "800", "fontSize": "2.2rem",
                                   "marginBottom": "4px"}),
                    html.P(
                        f"Classe {pred}  ·  Confiança: {probas[modelo.classes_.tolist().index(pred)] * 100:.1f}%",
                        className="text-secondary mb-0",
                        style={"fontSize": "0.82rem"},
                    ),
                ]),
                className="kpi-card mb-4",
                style={"border": f"2px solid {cor}",
                       "boxShadow": f"0 0 24px {cor}44"},
            ), md=12,
        )),
        dbc.Card(
            dbc.CardBody(dcc.Graph(figure=fig_bar, config={"displayModeBar": False})),
            className="kpi-card",
        ),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
