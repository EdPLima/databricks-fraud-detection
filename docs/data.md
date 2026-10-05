# Dados

## Fonte

Dataset: **IEEE-CIS Fraud Detection** (competicao Kaggle). Dados de transacoes financeiras com label de fraude, divididos em duas tabelas:

* **Transaction**: 590.540 linhas, 393 colunas. Informacoes da transacao (valor, cartao, endereco, email, C1-C14, D1-D15, M1-M9, V1-V339).
* **Identity**: 144.233 linhas, 41 colunas. Informacoes de identidade do dispositivo (DeviceType, DeviceInfo, id_01-id_38). Join por `TransactionID`.

Ingestao via notebooks `pipeline/bronze/ingest_transaction_data` e `pipeline/bronze/ingest_identity_data`. Join e limpeza em `pipeline/silver/ingest_data_complete`.

## Schema Silver

Tabela: `fraud_detection_dev.silver.fraud_data_clean`

590.540 linhas, 434 colunas. Principais grupos:

| Grupo | Colunas | Descricao |
|---|---|---|
| Chaves | `TransactionID`, `TransactionDT` | ID da transacao, timestamp em segundos |
| Label | `target` | 1 = fraude, 0 = legitima |
| Transacao | `TransactionAmt`, `card1-6`, `addr1-2`, `dist1-2`, `P_emaildomain`, `R_emaildomain` | Dados da transacao |
| Contadores | `C1-C14` | Contadores de atividade |
| Temporais | `D1-D15` | Deltas temporais |
| Categoricos | `M1-M9`, `ProductCD`, `DeviceType`, `DeviceInfo` | Flags categoricos |
| Anonimizados | `V1-V339` | Features anonimizadas pelo Vesta |
| Identidade | `id_01-id_38` | Informacoes de dispositivo e identidade |

## EDA

Notebook: `01_EDA`. Fonte dos dados: `fraud_detection_dev.silver.fraud_data_clean`.

### Distribuicao da variavel-alvo

| target | count | percentage |
|---|---|---|
| 0 (legitima) | 569.877 | 96,5% |
| 1 (fraude) | 20.663 | 3,5% |

Dataset altamente desbalanceado (ratio 27:1). Impacto direto no treino: uso de `class_weight` (peso inverso da proporcao) e otimizacao do threshold para maximizar F1.

### Qualidade dos dados

Analise de nulos, cardinalidade e variancia para todas as 434 colunas. Muitas features V possuem alta taxa de nulos (ate 86%). Nenhum tratamento de nulos na Silver; o preenchimento (imputer) ocorre no preprocessor do modelo.

### Outliers (Isolation Forest)

Isolation Forest com `contamination=0.05`, `n_estimators=300` aplicado em variaveis numericas. 5% das transacoes marcadas como anomalias.

Insight: **17,45% das anomalias sao fraudulentas** vs 3,5% na populacao geral. Os outliers nao foram removidos porque representam comportamentos naturais do fenomeno de fraude.

### Testes estatisticos

**KS (Kolmogorov-Smirnov)** para variaveis numericas. Top features por poder discriminante:

| KS | Feature | Interpretacao |
|---|---|---|
| 0.331 | C4 | Alto poder discriminante |
| 0.316 | C8 | Alto poder discriminante |
| 0.316 | V303 | Alto poder discriminante |
| 0.314 | V304 | Alto poder discriminante |
| 0.312 | V302 | Alto poder discriminante |
| 0.312 | C10 | Alto poder discriminante |

**Chi-Quadrado + V de Cramer** para variaveis categoricas. Top features por associacao com target:

| Chi2 | V de Cramer | Feature | Associacao |
|---|---|---|---|
| 72.289 | 0.350 | card1 | Forte |
| 44.514 | 0.275 | DeviceInfo | Moderada |
| 20.808 | 0.188 | id_31 | Moderada |
| 19.758 | 0.183 | id_30 | Moderada |

### Variaveis anonimizadas (V1-V339)

Information Value (IV) calculado para todas as 339 variaveis V. **19 variaveis com IV > 0.5** (poder preditivo alto).

Insight de variancia: **18 das 19 variaveis com IV > 0.5 apresentam razao de variancia > 3** entre fraudes e legitimas. Transacoes fraudulentas tem variancia muito maior. Exemplo extremo: V200 tem 200x mais variancia em fraudes. Isso indica que o modelo pode usar a variabilidade dessas features como sinal de fraude.

### Impacto financeiro

Analise do valor monetario de fraudes por segmento (card1, ProductCD, DeviceType). Concentracao de valor fraudado em poucos emissores de cartao.

### Dispositivo e identidade

Taxa de fraude por `DeviceType`: desktop e mobile com taxas diferentes. Combinacoes de `DeviceInfo` + `id_31` (browser) com maior concentracao de fraude.

## Preprocessamento

### Feature Selection (3 etapas)

Notebook: `02_FE_PRE_PROCE`. Aplicado no conjunto de treino para evitar data leak.

**Etapa 1: Qualidade**. Removidas 71 features com problemas estruturais:
* Variancia quase zero (var < 0.01)
* Cardinalidade problematica (constante ou ID-like: razao unico/nao-nulo > 95%)
* Quase constante (valor mais frequente em > 95% das linhas)

Resultado: 286 features restantes.

**Etapa 2: Evidencia Univariada**. Poder preditivo individual:
* Numericas: Information Value (IV) + KS
* Categoricas: Chi-Quadrado + V de Cramer
* Score unificado para ranking (KS e V de Cramer, escala 0-1)

Top features por evidencia: V258 (0.45), D5 (0.44), DeviceInfo (0.42), V257 (0.42), V265 (0.42).

**Etapa 3: Redundancia**. Remocao de features altamente correlacionadas:
* Numericas: correlacao de Pearson |r| > 0.85
* Categoricas: V de Cramer entre pares > 0.85
* Entre pares redundantes, remover a de menor evidencia univariada

Excecao: `DeviceType` foi restaurada apos remocao por fornecer informacao de origem (mobile/desktop).

### Preprocessor do Modelo

ColumnTransformer aplicado no treino, serializado com o modelo:

| Tipo | Pipeline | Features |
|---|---|---|
| Categoricas | SimpleImputer(most_frequent) + OrdinalEncoder(handle_unknown, unknown_value=-1) | 20 |
| Numericas | SimpleImputer(median) + StandardScaler() | 132 |

Total: 152 features processadas. O preprocessor e empacotado como artifact pickle no modelo pyfunc.

## Feature Engineering

Notebook: `02_FE_PRE_PROCE`. Funcao `apply_feature_engineering_spark()` em PySpark nativo (sem pandas).

### Features criadas

**Temporais** (a partir de `TransactionDT`):

| Feature | Calculo |
|---|---|
| `transaction_hour` | (TransactionDT / 3600) % 24 |
| `transaction_day` | (TransactionDT / 86400) % 7 |
| `is_morning` | 1 se 6 <= hour < 12 |
| `is_afternoon` | 1 se 12 <= hour < 18 |
| `is_night` | 1 se hour >= 18 ou hour < 6 |
| `is_weekend` | 1 se day >= 5 |

**Valor da transacao**:

| Feature | Calculo |
|---|---|
| `TransactionAmt_log` | log1p(TransactionAmt) |
| `TransactionAmt_round` | 1 se TransactionAmt for inteiro |

**Email**:

| Feature | Calculo |
|---|---|
| `email_match` | 1 se P_emaildomain == R_emaildomain |
| `P_email_free` | 1 se P_emaildomain in dominios gratuitos (gmail, yahoo, hotmail, etc) |
| `R_email_free` | 1 se R_emaildomain in dominios gratuitos |

**Agregacoes por cartao** (Window partitionBy `card1`):

| Feature | Calculo |
|---|---|
| `card1_tx_count` | count(*) over(card1) |
| `card1_amt_mean` | mean(TransactionAmt) over(card1) |
| `card1_amt_std` | stddev(TransactionAmt) over(card1) |
| `TransactionAmt_dev_card` | TransactionAmt - card1_amt_mean |
| `TransactionAmt_zscore_card` | dev / (std + 1e-6) |

**Agregacoes por endereco** (Window partitionBy `addr1`):

| Feature | Calculo |
|---|---|
| `addr1_tx_count` | count(*) over(addr1) |
| `addr1_amt_mean` | mean(TransactionAmt) over(addr1) |

**Device**:

| Feature | Calculo |
|---|---|
| `DeviceType_isnull` | 1 se DeviceType is null |

Total: 20 features criadas. Combinadas com as 132 features selecionadas da Silver, resultam em 152 features finais.

## Feature Table Gold

Tabela: `fraud_detection_dev.gold.fraud_features`

Feature table no Unity Catalog com `FeatureEngineeringClient.create_table()`:

* **Primary keys**: `TransactionID` + `TransactionDT` (timeseries column)
* **156 colunas**: 152 features + TransactionID + TransactionDT + target + split + class_weight
* **590.540 linhas**: train (357.753) + test (118.579) + val (114.208)
* **Conteudo**: features em forma RAW (sem encoding/scaling; preprocessamento fica no modelo pyfunc)
* **Target e split** permanecem na tabela para rastreabilidade do treino
* **Change Data Feed** habilitado para sincronizacao com Online Store

Metadados das features (descricoes) aplicados ao Unity Catalog para governanca e descobabilidade.