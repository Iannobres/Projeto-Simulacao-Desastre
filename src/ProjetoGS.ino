#include <wire.h>
#include <DHT.h>
#include <DHT_U.h>
#include <Ultrasonic.h>
#include <SHA256.h>
#include "config.h"

DHT dht(DHT_PIN, DHT_TYPE);
Ultrasonic ultrasonic(TRIG_PIN, ECHO_PIN);

// --- Tipos ---

struct LeituraSensores {
  float temp;
  float umid;
  float lux;
  int   dist;
  int   potenc;
  bool  valida;  // false se DHT retornou NaN
};

enum NivelRisco   { SEGURO, ALERTA, PERIGO };
enum EstadoSistema { NORMAL, AGUARDANDO_SENHA, BLOQUEADO, ADMIN };

// --- Estado global ---
EstadoSistema estado = NORMAL;
unsigned long previousMillis       = 0;
unsigned long adminUltimaAtividade = 0;
unsigned long bloqueioInicio       = 0;
int tentativasRestantes = ADMIN_MAX_TENTATIVAS;
int Contador = 1;

// ---------------------------------------------------------------------------
// Hash SHA256
// ---------------------------------------------------------------------------
String gerarHashSHA256(const String& entrada) {
  SHA256 sha256;
  sha256.reset();
  sha256.update((const uint8_t*)entrada.c_str(), entrada.length());
  uint8_t resultado[32];
  sha256.finalize(resultado, sizeof(resultado));
  String hex = "";
  for (int i = 0; i < 32; i++) {
    if (resultado[i] < 16) hex += "0";
    hex += String(resultado[i], HEX);
  }
  return hex;
}

// ---------------------------------------------------------------------------
// Leitura dos sensores
// ---------------------------------------------------------------------------
LeituraSensores lerSensores() {
  LeituraSensores s;
  s.temp   = dht.readTemperature();
  s.umid   = dht.readHumidity();
  s.dist   = ultrasonic.distanceRead();
  s.potenc = analogRead(POTENCIOMETRO);
  s.valida = !(isnan(s.temp) || isnan(s.umid));

  // LDR → lux com proteção contra divisão por zero
  float ldrADC    = (float)map(analogRead(LDR_PIN), 4095, 0, 1024, 0);
  float tensao    = ldrADC / 1024.0f * 5.0f;
  float denom     = 1.0f - tensao / 5.0f;
  float resist    = (fabsf(denom) < 1e-6f) ? 1e6f : (2000.0f * tensao / denom);
  s.lux = (float)pow(50e3 * pow(10.0, 0.7) / resist, 1.0 / 0.7);

  return s;
}

// ---------------------------------------------------------------------------
// Avaliação de risco
// ---------------------------------------------------------------------------
NivelRisco avaliarRisco(const LeituraSensores& s) {
  if (!s.valida) return PERIGO;  // falha de sensor = conservador

  bool perigo = false, alerta = false;

  // Temperatura
  if      (s.temp > TEMP_PERIGO_ALTO  || s.temp < TEMP_PERIGO_BAIXO)  perigo = true;
  else if (s.temp > TEMP_ALERTA_ALTO  || s.temp < TEMP_ALERTA_BAIXO)  alerta = true;

  // Umidade
  if      (s.umid > UMID_PERIGO_ALTO  || s.umid < UMID_PERIGO_BAIXO)  perigo = true;
  else if (s.umid > UMID_ALERTA_ALTO  || s.umid < UMID_ALERTA_BAIXO)  alerta = true;

  // Luminosidade
  if      (s.lux < LUX_PERIGO_BAIXO  || s.lux > LUX_PERIGO_ALTO)     perigo = true;
  else if (s.lux < LUX_ALERTA_BAIXO  || s.lux > LUX_ALERTA_ALTO)     alerta = true;

  // Nível da água
  if      (s.dist < DIST_PERIGO)   perigo = true;
  else if (s.dist < DIST_ALERTA)   alerta = true;

  // Vibração
  if      (s.potenc > VIBRACAO_PERIGO)   perigo = true;
  else if (s.potenc > VIBRACAO_ALERTA)   alerta = true;

  return perigo ? PERIGO : (alerta ? ALERTA : SEGURO);
}

// ---------------------------------------------------------------------------
// Texto descritivo de risco (para coluna Status Risco)
// ---------------------------------------------------------------------------
String textoRisco(const LeituraSensores& s, NivelRisco nivel) {
  if (!s.valida)      return "ERRO: falha no sensor DHT";
  if (nivel == SEGURO) return "SEGURO: Nenhum risco detectado";

  String riscos = "";

  // Temperatura
  if      (s.temp > TEMP_PERIGO_ALTO)   riscos += "Calor extremo; ";
  else if (s.temp > TEMP_ALERTA_ALTO)   riscos += "Calor alto (alerta); ";
  if      (s.temp < TEMP_PERIGO_BAIXO)  riscos += "Frio extremo; ";
  else if (s.temp < TEMP_ALERTA_BAIXO)  riscos += "Frio baixo (alerta); ";

  // Umidade
  if      (s.umid > UMID_PERIGO_ALTO)   riscos += "Umidade extrema; ";
  else if (s.umid > UMID_ALERTA_ALTO)   riscos += "Umidade alta (alerta); ";
  if      (s.umid < UMID_PERIGO_BAIXO)  riscos += "Umidade muito baixa; ";
  else if (s.umid < UMID_ALERTA_BAIXO)  riscos += "Umidade baixa (alerta); ";

  // Luminosidade
  if      (s.lux < LUX_PERIGO_BAIXO)   riscos += "Luminosidade extremamente baixa (fumaca densa/possivel incendio); ";
  else if (s.lux < LUX_ALERTA_BAIXO)   riscos += "Luminosidade baixa (fumaca moderada); ";
  if      (s.lux > LUX_PERIGO_ALTO)    riscos += "Luminosidade critica (possivel presenca de chamas); ";
  else if (s.lux > LUX_ALERTA_ALTO)    riscos += "Luminosidade alta (luz intensa, alerta); ";

  // Nível da água
  if      (s.dist < DIST_PERIGO)   riscos += "Enchente iminente; ";
  else if (s.dist < DIST_ALERTA)   riscos += "Nivel da agua elevado (alerta); ";

  // Vibração
  if      (s.potenc > VIBRACAO_PERIGO)   riscos += "Vibracao forte detectada! Possivel deslizamento; ";
  else if (s.potenc > VIBRACAO_ALERTA)   riscos += "Vibracao moderada (alerta); ";

  String prefixo = (nivel == PERIGO) ? "PERIGO: " : "ALERTA: ";
  return prefixo + riscos;
}

// ---------------------------------------------------------------------------
// Classificação ML (0–6) alinhada com as 7 classes do modelo treinado
//   0=Seguro  1=Calor/Umidade  2=Incêndio/Fumaça  3=Vibração Moderada
//   4=Deslizamento  5=Alerta Enchente  6=Enchente
// ---------------------------------------------------------------------------
int classificarDesastre(const LeituraSensores& s) {
  if (!s.valida) return 1;  // sensor inválido = alerta conservador
  if (s.dist < DIST_PERIGO)                                              return 6;
  if (s.potenc > VIBRACAO_PERIGO)                                         return 4;
  if (s.lux < LUX_PERIGO_BAIXO || s.lux > LUX_PERIGO_ALTO)              return 2;
  if (s.dist < DIST_ALERTA)                                              return 5;
  if (s.potenc > VIBRACAO_ALERTA)                                         return 3;
  if (s.temp > TEMP_PERIGO_ALTO  || s.temp < TEMP_PERIGO_BAIXO ||
      s.umid > UMID_PERIGO_ALTO  || s.umid < UMID_PERIGO_BAIXO)         return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// LEDs
// ---------------------------------------------------------------------------
void atualizarLEDs(NivelRisco nivel) {
  digitalWrite(LED_AMARELO,  nivel == ALERTA ? HIGH : LOW);
  digitalWrite(LED_VERMELHO, nivel == PERIGO ? HIGH : LOW);
  digitalWrite(LED_VERDE,    nivel == SEGURO ? HIGH : LOW);
}

// ---------------------------------------------------------------------------
// Formatação CSV — Modo Normal
// Colunas: Temperatura, Umidade, Luminosidade(bool), NivelAgua(bool), Vibracao(bool), StatusRisco
// ---------------------------------------------------------------------------
String formatarCSVNormal(const LeituraSensores& s, const String& riscoResumo) {
  if (!s.valida) return "NaN, NaN, 0, 0, 0, ERRO: falha no sensor DHT";

  bool luzAnomala       = (s.lux < LUX_ALERTA_BAIXO || s.lux > LUX_ALERTA_ALTO);
  bool nivelDetectado   = (s.dist < DIST_ALERTA);
  bool vibracaoAtiva    = (s.potenc > VIBRACAO_ALERTA);

  String csv = "";
  csv += String(s.temp, 2)    + ", ";
  csv += String((int)s.umid)  + ", ";
  csv += String(luzAnomala)   + ", ";
  csv += String(nivelDetectado) + ", ";
  csv += String(vibracaoAtiva)  + ", ";
  csv += riscoResumo;
  return csv;
}

// ---------------------------------------------------------------------------
// Formatação CSV — Modo Admin
// Colunas: Data, ID, Temperatura, Umidade, Luminosidade, NivelAgua, Vibracao, StatusRisco, SaidaML
// ---------------------------------------------------------------------------
String formatarCSVAdmin(const LeituraSensores& s, int id,
                        int day, int hour, int minute,
                        const String& riscoResumo, int labelML) {
  String csv = "Dia ";
  csv += String(day) + " ";
  if (hour < 10)   csv += "0";
  csv += String(hour) + ":";
  if (minute < 10) csv += "0";
  csv += String(minute) + ", ";
  csv += String(id) + ", ";
  csv += (s.valida ? String(s.temp, 2) : "NaN") + ", ";
  csv += (s.valida ? String((int)s.umid) : "NaN") + ", ";
  csv += String(s.lux)    + ", ";
  csv += String(s.dist)   + ", ";
  csv += String(s.potenc) + ", ";
  csv += riscoResumo      + ", ";
  csv += String(labelML);
  return csv;
}

// ---------------------------------------------------------------------------
// Timestamp simulado (cada tick = 30 minutos)
// ---------------------------------------------------------------------------
void calcularTempo(int contador, int& day, int& hour, int& minute) {
  unsigned long totalMinutes = (unsigned long)(contador - 1) * 30;
  minute = totalMinutes % 60;
  hour   = (totalMinutes / 60) % 24;
  day    = 1 + (int)(totalMinutes / (24 * 60));
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  dht.begin();
  pinMode(LDR_PIN,      INPUT);
  pinMode(POTENCIOMETRO, INPUT);
  pinMode(BOTAO_PIN,    INPUT_PULLUP);
  pinMode(LED_AMARELO,  OUTPUT);
  pinMode(LED_VERDE,    OUTPUT);
  pinMode(LED_VERMELHO, OUTPUT);
  Serial.println("Temperatura (oC), Umidade (%), Luminosidade (LUX), Nivel da agua, Vibracao do solo, Status Risco");
}

// ---------------------------------------------------------------------------
// Loop principal — máquina de estados não-bloqueante
// ---------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // --- BLOQUEADO: aguarda lockout expirar, pisca LEDs ---
  if (estado == BLOQUEADO) {
    bool ligado = ((now - bloqueioInicio) / 650) % 2 == 0;
    digitalWrite(LED_AMARELO,  ligado ? HIGH : LOW);
    digitalWrite(LED_VERDE,    ligado ? HIGH : LOW);
    digitalWrite(LED_VERMELHO, ligado ? HIGH : LOW);
    if (now - bloqueioInicio >= ADMIN_BLOQUEIO_MS) {
      tentativasRestantes = ADMIN_MAX_TENTATIVAS;
      estado = AGUARDANDO_SENHA;
      Serial.println("Voce pode tentar novamente. Digite a senha ou 'EXIT' para sair:");
    }
    return;
  }

  // --- AGUARDANDO_SENHA: valida entrada serial ---
  if (estado == AGUARDANDO_SENHA) {
    if (Serial.available()) {
      String entrada = Serial.readStringUntil('\n');
      entrada.trim();
      if (entrada.length() > 64) entrada = "";  // proteção contra overflow

      if (entrada == "EXIT") {
        estado = NORMAL;
        Serial.println("Saindo do modo admin...");
        Serial.println("Temperatura (oC), Umidade (%), Luminosidade (LUX), Nivel da agua, Vibracao do solo, Status Risco");
      } else if (gerarHashSHA256(entrada) == ADMIN_HASH) {
        estado = ADMIN;
        adminUltimaAtividade = now;
        tentativasRestantes  = ADMIN_MAX_TENTATIVAS;
        Serial.println("Acesso concedido ao modo admin!");
        Serial.println("Data, ID, Temperatura (oC), Umidade (%), Luminosidade (LUX), Nivel da agua, Vibracao do solo, Status Risco, Saida ML");
      } else {
        tentativasRestantes--;
        if (tentativasRestantes == 0) {
          bloqueioInicio = now;
          estado = BLOQUEADO;
          Serial.println("Muitas tentativas incorretas. Aguardando 30 segundos...");
        } else {
          Serial.print("Senha incorreta. Tentativas restantes: ");
          Serial.println(tentativasRestantes);
        }
      }
    }
    return;
  }

  // --- Tick de leitura (NORMAL e ADMIN) ---
  if (now - previousMillis < LOOP_INTERVAL_MS) return;
  previousMillis = now;

  // Botão → iniciar autenticação (apenas no modo NORMAL)
  if (estado == NORMAL && digitalRead(BOTAO_PIN) == LOW) {
    estado = AGUARDANDO_SENHA;
    tentativasRestantes = ADMIN_MAX_TENTATIVAS;
    Serial.println("Digite a senha ou 'EXIT' para sair:");
    return;
  }

  // Modo ADMIN: verifica comando EXIT e timeout por inatividade
  if (estado == ADMIN) {
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      adminUltimaAtividade = now;
      if (cmd == "EXIT") {
        estado = NORMAL;
        Serial.println("Saindo do modo admin...");
        Serial.println("Temperatura (oC), Umidade (%), Luminosidade (LUX), Nivel da agua, Vibracao do solo, Status Risco");
        return;
      }
    }
    if (now - adminUltimaAtividade > ADMIN_TIMEOUT_MS) {
      estado = NORMAL;
      Serial.println("Timeout: saindo do modo admin por inatividade.");
      Serial.println("Temperatura (oC), Umidade (%), Luminosidade (LUX), Nivel da agua, Vibracao do solo, Status Risco");
      return;
    }
  }

  // --- Leitura e saída ---
  LeituraSensores sensores = lerSensores();
  NivelRisco nivel         = avaliarRisco(sensores);
  String riscoResumo       = textoRisco(sensores, nivel);
  atualizarLEDs(nivel);

  if (estado == NORMAL) {
    Serial.println(formatarCSVNormal(sensores, riscoResumo));
  } else if (estado == ADMIN) {
    int day, hour, minute;
    calcularTempo(Contador, day, hour, minute);
    int labelML = classificarDesastre(sensores);
    Serial.println(formatarCSVAdmin(sensores, Contador, day, hour, minute, riscoResumo, labelML));
  }

  Contador++;
}
