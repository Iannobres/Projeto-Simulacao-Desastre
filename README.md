# Sistema de Previsão de Desastres Naturais — ESP32 + ML + Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PlatformIO](https://img.shields.io/badge/PlatformIO-ESP32-orange?logo=platformio)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f7931e?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

Sistema **end-to-end** de monitoramento de desastres naturais: firmware IoT em ESP32, pipeline de Machine Learning com Random Forest e dashboard interativo para visualização e previsão em tempo real.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ESP32 (Wokwi / Hardware)                      │
│  DHT22 · LDR · Ultrassônico · Potenciômetro · LEDs · Botão      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ CSV serial (115200 baud)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Pipeline de ML (Jupyter)                     │
│  Carrega Excel · Feature Engineering · Random Forest · joblib   │
│  Validação Cruzada · Feature Importance · Exporta .pkl          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ disaster_model.pkl
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Dashboard Streamlit                          │
│  Monitor de Sensores · Previsão ao vivo · Insights do modelo    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Funcionalidades

### Firmware ESP32 (`src/`)
- **Modo Normal:** lê sensores e envia CSV compacto com flags booleanas e status de risco
- **Modo Admin:** autenticação SHA256, envia CSV completo com label ML multiclasse (0–6)
- Máquina de estados não-bloqueante: sem `delay()` no loop principal
- Timeout automático do modo admin (5 min), bloqueio por tentativas incorretas
- Validação de NaN no DHT22, proteção contra divisão por zero no LDR

### Sensores Simulados

| Sensor          | Pino | Finalidade                               |
|-----------------|------|------------------------------------------|
| DHT22           | 16   | Temperatura e umidade do ar              |
| LDR             | 4    | Luminosidade — detecção de fumaça/chamas |
| Ultrassônico    | 5/18 | Nível da água — enchentes               |
| Potenciômetro   | 27   | Vibração do solo — deslizamentos         |

**LEDs:** 🟢 Verde (seguro) · 🟡 Amarelo (alerta) · 🔴 Vermelho (perigo)

### Pipeline de ML (`ml/`)
- Dataset: 2.457 leituras simuladas (52 dias), 8 features
- Random Forest com `class_weight='balanced'` (corrige desbalanceamento)
- Validação cruzada estratificada 5-Fold com F1-Macro
- Feature importance, análise de falsos negativos
- Exportação do modelo treinado com joblib
- **Acurácia: ~85%** no conjunto de teste

**7 classes de saída:**
| Classe | Descrição             |
|--------|-----------------------|
| 0      | Sem desastre          |
| 1      | Alerta de incêndio    |
| 2      | Incêndio / fumaça     |
| 3      | Alerta de deslizamento|
| 4      | Deslizamento          |
| 5      | Alerta de enchente    |
| 6      | Enchente              |

### Dashboard Streamlit (`dashboard/`)
- **Aba 1 — Monitor:** gráficos de série temporal por sensor, métricas de risco
- **Aba 2 — Previsão:** upload de CSV ou entrada manual de valores → previsão + probabilidades
- **Aba 3 — Insights:** feature importance, distribuição de classes, parâmetros do modelo

---

## Estrutura do Projeto

```
Projeto-Simulacao-Desastre/
├── src/
│   ├── ProjetoGS.ino       # Firmware ESP32 (refatorado, modular)
│   └── config.h            # Limiares e constantes configuráveis
├── ml/
│   ├── Modelo_Previsão_de_Desastre.ipynb  # Pipeline ML completo
│   ├── data/
│   │   └── Dados ESP32 GS.xlsx            # Dataset de treino (2.457 linhas)
│   ├── models/             # Gerado ao rodar o notebook
│   │   └── disaster_model.pkl
│   └── requirements.txt
├── dashboard/
│   ├── app.py              # Dashboard Streamlit (3 abas)
│   └── requirements.txt
├── diagram.json            # Esquema Wokwi (ESP32 + sensores)
├── platformio.ini          # Dependências do firmware
└── wokwi.toml              # Configuração da simulação
```

---

## Como Rodar

### 1. Firmware (Simulação Wokwi)

```bash
# Instalar PlatformIO CLI ou extensão VS Code

# Compilar firmware
pio run

# Monitorar saída serial
pio device monitor --baud 115200
```

Abra o VS Code com a extensão Wokwi e o arquivo `wokwi.toml` para simular o circuito.

### 2. Notebook ML

```bash
cd ml
pip install -r requirements.txt
jupyter notebook "Modelo_Previsão_de_Desastre.ipynb"
```

Execute todas as células. O modelo será salvo em `ml/models/disaster_model.pkl`.

### 3. Dashboard Streamlit

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

---

## Resultados do Modelo

| Classe               | Precisão | Recall | F1-Score |
|----------------------|----------|--------|----------|
| Sem Desastre (0)     | ~0.97    | ~0.90  | ~0.93    |
| Alerta Incêndio (1)  | ~0.96    | ~0.96  | ~0.96    |
| Incêndio (2)         | ~0.99    | ~0.99  | ~0.99    |
| Alerta Desliz. (3)   | ~1.00    | ~1.00  | ~1.00    |
| Deslizamento (4)     | ~1.00    | ~1.00  | ~1.00    |
| Alerta Enchente (5)  | ~0.50    | ~0.33  | ~0.40    |
| Enchente (6)         | ~1.00    | ~1.00  | ~1.00    |

> Classe 5 tem baixa representatividade no dataset (~16 amostras de teste). `class_weight='balanced'` mitiga o problema.

---

## Autores

**Ian Nobres Azevedo** e **Matheus Mikio Tutume Lourenço**

Projeto desenvolvido como parte de estudos em IoT e Machine Learning.
