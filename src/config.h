#pragma once

// --- Pinos ---
#define DHT_PIN         16
#define LED_VERMELHO    23
#define LED_AMARELO     22
#define LED_VERDE       21
#define LDR_PIN         4
#define BOTAO_PIN       17
#define TRIG_PIN        5
#define ECHO_PIN        18
#define POTENCIOMETRO   27
#define DHT_TYPE        DHT22

// --- Limiares de temperatura (°C) ---
#define TEMP_PERIGO_ALTO    45.0f
#define TEMP_ALERTA_ALTO    35.0f
#define TEMP_PERIGO_BAIXO   -5.0f
#define TEMP_ALERTA_BAIXO   15.0f

// --- Limiares de umidade (%) ---
#define UMID_PERIGO_ALTO    90.0f
#define UMID_ALERTA_ALTO    70.0f
#define UMID_PERIGO_BAIXO   10.0f
#define UMID_ALERTA_BAIXO   30.0f

// --- Limiares de luminosidade (lux) ---
#define LUX_PERIGO_BAIXO    100.0f
#define LUX_ALERTA_BAIXO    500.0f
#define LUX_ALERTA_ALTO     2000.0f
#define LUX_PERIGO_ALTO     8000.0f

// --- Limiares de distância / nível da água (cm) ---
#define DIST_PERIGO     50
#define DIST_ALERTA     100

// --- Limiares de vibração (ADC) ---
#define VIBRACAO_PERIGO     4000
#define VIBRACAO_ALERTA     3000

// --- Autenticação admin ---
// Senha: "ADMINGS"
#define ADMIN_HASH              "6c0db579efe383b1ec2a86c95338a6e1b4c874378434f6b86d26d9f4808c5e05"
#define ADMIN_MAX_TENTATIVAS    3
#define ADMIN_BLOQUEIO_MS       30000UL
#define ADMIN_TIMEOUT_MS        300000UL   // 5 minutos sem atividade

// --- Timing ---
#define LOOP_INTERVAL_MS        50UL       // 50 ms = 30 minutos simulados por tick
