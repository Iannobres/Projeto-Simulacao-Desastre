import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Configuração da página ---
st.set_page_config(
    page_title="Sistema de Previsão de Desastres",
    page_icon="🌪️",
    layout="wide",
)

# --- Caminhos ---
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "ml" / "models" / "disaster_model.pkl"
DATA_PATH  = ROOT / "ml" / "data" / "Dados ESP32 GS.xlsx"

# --- Constantes ---
CLASSES = {
    0: ("Sem Desastre",          "🟢", "#2ecc71"),
    1: ("Alerta de Incêndio",    "🟡", "#f39c12"),
    2: ("Incêndio",              "🔴", "#e74c3c"),
    3: ("Alerta de Deslizamento","🟡", "#f39c12"),
    4: ("Deslizamento",          "🔴", "#e74c3c"),
    5: ("Alerta de Enchente",    "🟡", "#f39c12"),
    6: ("Enchente",              "🔴", "#e74c3c"),
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

SENSOR_COLS = ["Temperatura (ºC)", "Umidade (%)", "Luminosidade (LUX)", "Nivel da agua", "Vibração do solo"]


# --- Funções auxiliares ---
@st.cache_resource
def carregar_modelo():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def carregar_dados_demo():
    if not DATA_PATH.exists():
        return None
    df = pd.read_excel(DATA_PATH)
    df = _engenharia_temporal(df)
    return df


def _engenharia_temporal(df: pd.DataFrame) -> pd.DataFrame:
    if "Data" not in df.columns:
        return df
    df = df.copy()
    df["Dia"]    = df["Data"].str.extract(r"Dia (\d+)").astype(int)
    df["Hora"]   = df["Data"].str.extract(r"(\d+):(\d+)")[0].astype(int)
    df["Minuto"] = df["Data"].str.extract(r"(\d+):(\d+)")[1].astype(int)
    df["Tempo_Minutos"] = (df["Dia"] - 1) * 24 * 60 + df["Hora"] * 60 + df["Minuto"]
    hora_dia_min = df["Hora"] * 60 + df["Minuto"]
    df["Hora_sin"] = np.sin(2 * np.pi * hora_dia_min / 1440)
    df["Hora_cos"] = np.cos(2 * np.pi * hora_dia_min / 1440)
    return df


def badge_classe(classe: int) -> str:
    nome, icone, _ = CLASSES[classe]
    return f"{icone} {nome}"


# --- Layout ---
st.title("🌪️ Sistema de Previsão de Desastres Naturais")
st.caption("ESP32 IoT + Machine Learning — Monitoramento Ambiental em Tempo Real")
st.divider()

modelo = carregar_modelo()
df_demo = carregar_dados_demo()

aba1, aba2, aba3 = st.tabs(["📡 Monitor de Sensores", "🔮 Previsão", "📊 Insights do Modelo"])

# =========================================================================
# Aba 1 — Monitor de Sensores
# =========================================================================
with aba1:
    st.subheader("Monitoramento de Dados dos Sensores")

    arquivo_csv = st.file_uploader(
        "Carregar CSV do ESP32 (modo admin)",
        type=["csv"],
        help="Arquivo no formato: Data, ID, Temperatura, Umidade, Luminosidade, Nivel da agua, Vibracao, Status Risco, Saida ML",
    )

    if arquivo_csv is not None:
        df_sensor = pd.read_csv(arquivo_csv)
        df_sensor.columns = df_sensor.columns.str.strip()
    elif df_demo is not None:
        st.info("Exibindo dados de demonstração (Dados ESP32 GS.xlsx). Faça upload de um CSV para usar dados reais.")
        df_sensor = df_demo
    else:
        st.warning("Modelo ou dados não encontrados. Execute o notebook ML primeiro.")
        st.stop()

    # Métricas resumidas
    col1, col2, col3, col4 = st.columns(4)
    if "Saída ML" in df_sensor.columns:
        pct_risco = df_sensor["Saída ML"].mean() * 100
        col1.metric("Total de Leituras", f"{len(df_sensor):,}")
        col2.metric("Leituras com Risco", f"{df_sensor['Saída ML'].sum():,}", f"{pct_risco:.1f}%")
        if "Temperatura (ºC)" in df_sensor.columns:
            col3.metric("Temp. Máxima", f"{df_sensor['Temperatura (ºC)'].max():.1f} °C")
        if "Nivel da agua" in df_sensor.columns:
            col4.metric("Nível Máx. Água", f"{df_sensor['Nivel da agua'].max()} cm")

    # Gráficos de sensores
    sensores_presentes = [c for c in SENSOR_COLS if c in df_sensor.columns]
    eixo_x = "Tempo_Minutos" if "Tempo_Minutos" in df_sensor.columns else df_sensor.index.name or df_sensor.columns[0]

    for sensor in sensores_presentes:
        fig = px.line(
            df_sensor,
            x=eixo_x,
            y=sensor,
            title=sensor,
            labels={eixo_x: "Tempo (minutos)", sensor: sensor},
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# Aba 2 — Previsão
# =========================================================================
with aba2:
    st.subheader("Previsão de Desastres")

    if modelo is None:
        st.error("Modelo não encontrado. Execute o notebook ML e rode a célula de exportação (joblib) para gerar `ml/models/disaster_model.pkl`.")
        st.stop()

    modo = st.radio("Modo de entrada", ["Upload de CSV", "Entrada manual"], horizontal=True)

    if modo == "Upload de CSV":
        arquivo_pred = st.file_uploader("CSV com colunas de sensor (sem coluna de risco)", type=["csv"], key="pred")
        if arquivo_pred is not None:
            df_pred_raw = pd.read_csv(arquivo_pred)
            df_pred_raw.columns = df_pred_raw.columns.str.strip()
            df_pred = _engenharia_temporal(df_pred_raw)

            features_ok = all(f in df_pred.columns for f in FEATURES)
            if not features_ok:
                st.warning(f"Colunas esperadas: {FEATURES}")
                st.write("Colunas encontradas:", df_pred.columns.tolist())
            else:
                X_pred = df_pred[FEATURES]
                preds = modelo.predict(X_pred)
                probas = modelo.predict_proba(X_pred)

                df_pred["Previsão"] = [badge_classe(p) for p in preds]
                st.dataframe(
                    df_pred[[c for c in df_pred.columns if c in SENSOR_COLS + ["Previsão"]]],
                    use_container_width=True,
                )

                st.subheader("Distribuição das Previsões")
                contagem = pd.Series(preds).map({k: v[0] for k, v in CLASSES.items()}).value_counts()
                fig_pizza = px.pie(values=contagem.values, names=contagem.index, title="Classes previstas")
                st.plotly_chart(fig_pizza, use_container_width=True)

    else:  # Entrada manual
        st.markdown("Ajuste os valores dos sensores:")
        c1, c2 = st.columns(2)
        with c1:
            temp   = st.slider("Temperatura (°C)", -10.0, 90.0, 25.0, 0.5)
            umid   = st.slider("Umidade (%)", 0, 100, 50)
            lux    = st.slider("Luminosidade (LUX)", 0, 25000, 1200)
        with c2:
            nivel  = st.slider("Nível da Água (cm)", 0, 400, 200)
            vibr   = st.slider("Vibração do Solo (ADC)", 2000, 4095, 2500)
            hora   = st.slider("Hora do dia", 0, 23, 12)
            minuto = st.selectbox("Minuto", [0, 30])

        hora_dia_min = hora * 60 + minuto
        hora_sin = float(np.sin(2 * np.pi * hora_dia_min / 1440))
        hora_cos = float(np.cos(2 * np.pi * hora_dia_min / 1440))
        tempo_min = hora * 60 + minuto

        X_manual = pd.DataFrame([[temp, umid, lux, nivel, vibr, tempo_min, hora_sin, hora_cos]], columns=FEATURES)

        if st.button("🔮 Prever", type="primary"):
            pred = modelo.predict(X_manual)[0]
            probas_manual = modelo.predict_proba(X_manual)[0]
            nome, icone, cor = CLASSES[pred]

            st.markdown(f"## {icone} {nome}")

            df_probas = pd.DataFrame({
                "Classe": [CLASSES[c][0] for c in modelo.classes_],
                "Probabilidade": probas_manual,
            }).sort_values("Probabilidade", ascending=False)

            fig_bar = px.bar(
                df_probas, x="Probabilidade", y="Classe",
                orientation="h", title="Probabilidade por Classe",
                range_x=[0, 1],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================================
# Aba 3 — Insights do Modelo
# =========================================================================
with aba3:
    st.subheader("Insights do Modelo de Machine Learning")

    if modelo is None:
        st.error("Modelo não encontrado. Execute o notebook ML primeiro.")
        st.stop()

    # Feature Importance
    st.markdown("### Importância das Features")
    importances = modelo.feature_importances_
    df_imp = pd.DataFrame({"Feature": FEATURES, "Importância": importances})
    df_imp = df_imp.sort_values("Importância", ascending=True)

    fig_imp = px.bar(
        df_imp, x="Importância", y="Feature",
        orientation="h",
        title="Quais sensores mais influenciam as previsões?",
        color="Importância",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Distribuição do dataset
    if df_demo is not None and "Tipo_Desastre" in df_demo.columns:
        st.markdown("### Distribuição de Classes no Dataset de Treino")
        contagem_classes = df_demo["Tipo_Desastre"].map({k: v[0] for k, v in CLASSES.items()}).value_counts()
        fig_dist = px.pie(
            values=contagem_classes.values,
            names=contagem_classes.index,
            title="Distribuição das 7 classes (2.457 amostras)",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    elif df_demo is not None and "Saída ML" in df_demo.columns:
        st.markdown("### Distribuição do Dataset de Treino")
        contagem_bin = df_demo["Saída ML"].map({0: "Sem Risco", 1: "Com Risco"}).value_counts()
        fig_bin = px.pie(values=contagem_bin.values, names=contagem_bin.index, title="Seguro vs. Risco")
        st.plotly_chart(fig_bin, use_container_width=True)

    # Informações do modelo
    st.markdown("### Informações do Modelo")
    info = {
        "Algoritmo": "Random Forest Classifier",
        "Estimadores": modelo.n_estimators,
        "Class Weight": str(modelo.class_weight),
        "Classes": len(modelo.classes_),
        "Features": len(FEATURES),
    }
    st.table(pd.DataFrame(info.items(), columns=["Parâmetro", "Valor"]))
