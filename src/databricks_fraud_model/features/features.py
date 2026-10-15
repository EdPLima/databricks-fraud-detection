"""
Single source of truth para as 152 features do modelo XGBoost.

Esta lista e usada por:
- scoring_model (pipeline/scoring/) — seleciona features para enviar ao endpoint
- 04_DEPLOYED_MODEL (notebooks/) — define a assinatura do modelo pyfunc
- monitoring (futuro) — compara distribuicao das features treino vs producao
"""

MODEL_FEATURES = [
    'TransactionAmt', 'card2', 'card3', 'card5', 'addr1', 'dist1',
    'C4', 'C5', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15',
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
    'id_01', 'id_02', 'id_05', 'id_06', 'id_11', 'id_13', 'id_17',
    'id_19', 'id_20',
    'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_12', 'id_16', 'id_31', 'id_36', 'id_37', 'id_38',
    'DeviceType', 'DeviceInfo',
    'DeviceType_isnull', 'TransactionAmt_log', 'TransactionAmt_round',
    'transaction_hour', 'transaction_day', 'is_morning', 'is_afternoon',
    'is_night', 'is_weekend', 'card1_tx_count', 'card1_amt_mean',
    'card1_amt_std', 'TransactionAmt_dev_card', 'TransactionAmt_zscore_card',
    'device_mismatch', 'addr1_tx_count', 'addr1_amt_mean',
    'email_match', 'P_email_free', 'R_email_free'
]

assert len(MODEL_FEATURES) == 152, f"Expected 152 features, got {len(MODEL_FEATURES)}"
