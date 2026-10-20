"""
Definicoes compartilhadas das features do modelo de fraude.

Contrato de features versionado — toda mudanca em MODEL_FEATURES exige
incremento de FEATURE_CONTRACT_VERSION para garantir reprodutibilidade.

Este arquivo e a SINGLE SOURCE OF TRUTH para a lista de features que o
modelo XGBoost espera receber. Todos os notebooks e jobs de ETL, scoring
e retreino consultam MODEL_FEATURES para saber quais colunas processar.

Versionamento:
    FEATURE_CONTRACT_VERSION — string semantica (ex: "1.0.0")
    FEATURE_CONTRACT_HASH    — hash SHA-256 (16 chars) para deteccao de mudancas

Para adicionar/remover uma feature:
    1. Modificar MODEL_FEATURES
    2. Incrementar FEATURE_CONTRACT_VERSION
    3. Atualizar o assert (len == N)
    4. Rerrodar os testes (test_feature_contract.py)
"""

import hashlib  # Para calcular o hash do contrato de features

# ============================================================
# VERSIONAMENTO DO CONTRATO DE FEATURES
# ============================================================
# Incrementar a cada mudanca em MODEL_FEATURES.
# Formato semantico: MAJOR.MINOR.PATCH
#   MAJOR: remocao ou troca de features (breaking change)
#   MINOR: adicao de novas features (backward compatible)
#   PATCH: correcao de nomes sem mudar o conjunto
FEATURE_CONTRACT_VERSION = "1.0.0"

# ============================================================
# LISTA CANONICA DE FEATURES DO MODELO (152 features)
# ============================================================
# Esta lista define EXATAMENTE quais colunas o modelo XGBoost espera
# receber no payload REST. A ordem nao importa (o pyfunc seleciona por
# nome), mas o conjunto deve ser exato.
#
# Categorias de features:
#   - TransactionAmt, card*, addr*, dist1     — features originais do IEEE-CIS
#   - C4, C5, D2-D5, D10, D11, D15            — contadores e deltas temporais
#   - V3-V318 (subset)                        — features V (Vesta engineered)
#   - id_01-id_38 (subset)                    — features de identidade
#   - ProductCD, card4, card6, email domains  — features categoricas
#   - M2-M9                                   — features de matching
#   - DeviceType, DeviceInfo                  — features de dispositivo
#   - *_isnull, *_log, *_round, transaction_*  — features engenheiradas
#   - card1_tx_count, card1_amt_*             — agregacoes historicas
#   - email_match, *_email_free               — features de email engenheiradas

MODEL_FEATURES = [
    # --- Features originais numericas (IEEE-CIS transaction table) ---
    'TransactionAmt', 'card2', 'card3', 'card5', 'addr1', 'dist1',
    # --- Contadores C e deltas D (Vesta engineered) ---
    'C4', 'C5', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15',
    # --- Features V (subset selecionado por qualidade + evidencia) ---
    'V3', 'V5', 'V7', 'V13', 'V20', 'V35', 'V38', 'V40', 'V45',
    'V52', 'V53', 'V56', 'V62', 'V76', 'V77', 'V78', 'V83', 'V87',
    'V91', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102',
    'V103', 'V105', 'V106', 'V124', 'V126', 'V130', 'V132', 'V133',
    'V134', 'V136', 'V168', 'V169', 'V171', 'V174', 'V175', 'V185',
    'V189', 'V197', 'V198', 'V201', 'V209', 'V210', 'V217', 'V220',
    'V222', 'V223', 'V226', 'V228', 'V236', 'V237', 'V239', 'V244',
    'V251', 'V257', 'V258', 'V260', 'V263', 'V265', 'V271', 'V273',
    'V275', 'V277', 'V279', 'V282', 'V283', 'V284', 'V285', 'V287',
    'V289', 'V290', 'V294', 'V298', 'V299', 'V303', 'V306', 'V309',
    'V310', 'V312', 'V315', 'V316', 'V318',
    # --- Features de identidade (IEEE-CIS identity table, subset) ---
    'id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_13', 'id_17',
    'id_19', 'id_20',
    # --- Features categoricas (serao OrdinalEncoder no pyfunc) ---
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_16', 'id_31', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo',
    # --- Features engenheiradas (apply_feature_engineering_spark) ---
    'DeviceType_isnull', 'TransactionAmt_log', 'TransactionAmt_round',  # dispositivo e valor
    'transaction_hour', 'transaction_day',                            # temporais
    'is_morning', 'is_afternoon', 'is_night', 'is_weekend',           # periodos do dia
    'card1_tx_count', 'card1_amt_mean',                               # agregacoes historicas card1
    'card1_amt_std', 'TransactionAmt_dev_card', 'TransactionAmt_zscore_card',
    'device_mismatch', 'addr1_tx_count', 'addr1_amt_mean',            # agregacoes historicas addr1
    'email_match', 'P_email_free', 'R_email_free'                     # features de email
]

# Assert de integridade: falha imediatamente se o numero de features mudar
# sem atualizar este assert. Previne silenciosamente enviar features erradas ao modelo.
assert len(MODEL_FEATURES) == 152, (f"Expected 152 features, got {len(MODEL_FEATURES)}")

# ============================================================
# HASH DO CONTRATO PARA VERIFICACAO DE INTEGRIDADE
# ============================================================
# O hash e deterministico: se a lista de features mudar, o hash muda.
# Usado para verificar que o scoring e o treino usam o mesmo contrato.
# Logado em cada predicao (feature_contract_version + hash na tabela Gold).
FEATURE_CONTRACT_HASH = hashlib.sha256(
    ",".join(MODEL_FEATURES).encode()
).hexdigest()[:16]


def validate_features(columns, model_features=None):
    """
    Valida se um conjunto de colunas contem todas as features do contrato.

    Usado nos notebooks de ETL (Gold) e scoring para garantir que o DataFrame
    tem todas as 152 features que o modelo espera, antes de enviar ao endpoint.

    Args:
        columns:        lista de colunas disponiveis (ex: df.columns).
        model_features: lista esperada. Default: MODEL_FEATURES (152).

    Returns:
        tuple (bool, list):
            - True se todas as features estao presentes, False caso contrario
            - Lista de features faltantes (vazia se True)

    Exemplo:
        valid, missing = validate_features(df.columns)
        if not valid:
            raise ValueError(f"Features faltantes: {missing}")
    """
    if model_features is None:
        model_features = MODEL_FEATURES
    available = set(columns)
    missing = [f for f in model_features if f not in available]
    return (len(missing) == 0, missing)

# ============================================================
# DOMINIOS DE EMAIL GRATUITOS
# ============================================================
# Usado por apply_feature_engineering_spark() para criar as features
# binarias P_email_free e R_email_free (1 se dominio e gratuito).
# Adicionar novos dominios aqui se necessario (ex: protonmail.com).
FREE_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
}
