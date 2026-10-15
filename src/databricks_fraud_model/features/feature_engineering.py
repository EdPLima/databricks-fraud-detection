"""
Feature engineering compartilhado para o pipeline de deteccao de fraude.

Contem as listas de feature selection e a funcao de feature engineering
em PySpark usadas por ambos os notebooks Gold:
  - pipeline/gold/feature_engineering (treino)
  - pipeline/gold/feature_engineering_predict (predicao)
"""

from pyspark.sql import functions as F, Window

# ============================================================================
# Feature Selection — listas hardcoded do pipeline de experimentacao
# ============================================================================
# Etapa 1: Qualidade (71 removidas)
# Etapa 2: Evidencia univariada (calculada)
# Etapa 3: Redundancia (134 removidas, DeviceType recuperado)

REMOVED_QUALITY = [
    'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'C13', 'C14', 'D1', 'D10', 'D11', 'D12', 'D13', 'D14', 'D6', 'D7', 'D8',
    'D9', 'M1', 'M2', 'M3', 'V1', 'V10', 'V11', 'V138', 'V14', 'V140', 'V141',
    'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V15',
    'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158',
    'V159', 'V16', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V17',
    'V18', 'V19', 'V2', 'V3', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326',
    'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335',
    'V336', 'V337', 'V338', 'V339', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'id_01', 'id_02',
]

REMOVED_REDUNDANCY = [
    'C3', 'D15', 'D2', 'D3', 'D4', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108',
    'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117',
    'V118', 'V119', 'V12', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125',
    'V126', 'V127', 'V128', 'V129', 'V13', 'V130', 'V131', 'V132', 'V133',
    'V134', 'V135', 'V136', 'V137', 'V139', 'V166', 'V167', 'V168', 'V169',
    'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178',
    'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186', 'V187',
    'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196',
    'V197', 'V198', 'V199', 'V20', 'V200', 'V201', 'V202', 'V203', 'V204',
    'V205', 'V206', 'V207', 'V208', 'V209', 'V21', 'V210', 'V211', 'V212',
    'V213', 'V214', 'V215', 'V216', 'V217', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35',
    'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V68',
    'V97', 'V98', 'V99', 'id_12',
]

# Features do modelo que NAO devem ser dropadas (presentes no treino gold)
# Estas 55 colunas foram incorretamente incluidas em REMOVED_QUALITY/REMOVED_REDUNDANCY
# mas o modelo foi treinado com elas — devem ser preservadas no pipeline prd
MODEL_KEEP = {
    'C4', 'C5', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15',
    'V3', 'V5', 'V7', 'V13', 'V20', 'V35', 'V38', 'V40',
    'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V105', 'V106',
    'V124', 'V126', 'V130', 'V132', 'V133', 'V134', 'V136',
    'V168', 'V169', 'V171', 'V174', 'V175', 'V185', 'V189',
    'V197', 'V198', 'V201', 'V209', 'V210', 'V217',
    'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_01', 'id_02', 'id_12',
}

# Lista completa de features a remover (usar com df.drop(*FEATURES_DROP))
# Filtra MODEL_KEEP para nao dropar features que o modelo usa
FEATURES_DROP = [f for f in (REMOVED_QUALITY + REMOVED_REDUNDANCY + ['ingestion_timestamp']) if f not in MODEL_KEEP]


# ============================================================================
# Feature Engineering — funcao PySpark
# ============================================================================

FREE_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
    'live.com', 'icloud.com', 'aol.com', 'msn.com',
]


def apply_feature_engineering_spark(df):
    """
    Feature Engineering nativo em PySpark para producao.

    Features criadas:
    - Temporais: hora, dia, periodos (manha/tarde/noite), fim de semana
    - Transacao: log(valor), arredondamento
    - Email: match entre P/R, dominios gratuitos
    - Agregacoes: contagens e estatisticas por card1 e addr1
    - Device: flag de nulo
    """
    # Device
    df = df.withColumn('DeviceType_isnull', F.when(F.col('DeviceType').isNull(), 1).otherwise(0))

    # Transacao
    df = df.withColumn('TransactionAmt_log', F.log1p(F.col('TransactionAmt')))
    df = df.withColumn('TransactionAmt_round', F.when(F.col('TransactionAmt') % 1 == 0, 1).otherwise(0))

    # Temporais
    df = df.withColumn('transaction_hour', ((F.col('TransactionDT') / 3600) % 24).cast('int'))
    df = df.withColumn('transaction_day', ((F.col('TransactionDT') / 86400) % 7).cast('int'))
    df = df.withColumn('is_morning', F.when((F.col('transaction_hour') >= 6) & (F.col('transaction_hour') < 12), 1).otherwise(0))
    df = df.withColumn('is_afternoon', F.when((F.col('transaction_hour') >= 12) & (F.col('transaction_hour') < 18), 1).otherwise(0))
    df = df.withColumn('is_night', F.when((F.col('transaction_hour') >= 18) | (F.col('transaction_hour') < 6), 1).otherwise(0))
    df = df.withColumn('is_weekend', F.when(F.col('transaction_day') >= 5, 1).otherwise(0))

    # Email
    df = df.withColumn('email_match', F.when(F.col('P_emaildomain') == F.col('R_emaildomain'), 1).otherwise(0))
    df = df.withColumn('P_email_free', F.when(F.col('P_emaildomain').isin(FREE_DOMAINS), 1).otherwise(0))
    df = df.withColumn('R_email_free', F.when(F.col('R_emaildomain').isin(FREE_DOMAINS), 1).otherwise(0))

    # Agregacoes por card1
    w_card1 = Window.partitionBy('card1')
    df = df.withColumn('card1_tx_count', F.count('*').over(w_card1))
    df = df.withColumn('card1_amt_mean', F.mean('TransactionAmt').over(w_card1))
    df = df.withColumn('card1_amt_std', F.stddev('TransactionAmt').over(w_card1))
    df = df.withColumn('TransactionAmt_dev_card', F.col('TransactionAmt') - F.col('card1_amt_mean'))
    df = df.withColumn('TransactionAmt_zscore_card', F.col('TransactionAmt_dev_card') / (F.col('card1_amt_std') + F.lit(1e-6)))

    # Agregacoes por addr1
    w_addr1 = Window.partitionBy('addr1')
    df = df.withColumn('addr1_tx_count', F.count('*').over(w_addr1))
    df = df.withColumn('addr1_amt_mean', F.mean('TransactionAmt').over(w_addr1))

    # Device mismatch (placeholder)
    df = df.withColumn('device_mismatch', F.lit(0))

    return df