import pytest

from databricks_fraud_model.features import (
    FREE_DOMAINS,
    MODEL_FEATURES,
    apply_feature_engineering_spark,
)

# ============================================================
# 1. VALIDAÇÃO DAS FEATURES DO MODELO
# ============================================================

def test_model_features_count():
    """Verifica a quantidade esperada de features."""
    assert len(MODEL_FEATURES) == 152


def test_model_features_no_duplicates():
    """Garante que não existem features duplicadas."""
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


def test_free_domains_is_set():
    """Valida a configuração dos domínios gratuitos."""
    assert isinstance(FREE_DOMAINS, set)
    assert "gmail.com" in FREE_DOMAINS
    assert len(FREE_DOMAINS) > 0


# ============================================================
# 2. FIXTURE DE TRANSAÇÕES CONTROLADAS
# ============================================================

@pytest.fixture
def temporal_transactions(spark):
    """
    Cria transações com histórico conhecido.

    Cartão 1234:
      TX001 -> t=100, valor=100
      TX002 -> t=200, valor=200
      TX003 -> t=300, valor=500

    Cartão 5678:
      TX004 -> t=400, valor=300

    Os valores permitem verificar manualmente
    contagens, médias e ausência de vazamento.
    """

    data = [
        (
            "TX001", 100, 100.0, 1234, 101,
            "mobile", "gmail.com", "gmail.com"
        ),
        (
            "TX002", 200, 200.0, 1234, 101,
            None, "gmail.com", "yahoo.com"
        ),
        (
            "TX003", 300, 500.0, 1234, 101,
            "desktop", "gmail.com", None
        ),
        (
            "TX004", 400, 300.0, 5678, 202,
            "mobile", "hotmail.com", "hotmail.com"
        ),
    ]

    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "DeviceType",
        "P_emaildomain",
        "R_emaildomain",
    ]

    return spark.createDataFrame(data, columns)


@pytest.fixture
def engineered_rows(temporal_transactions):
    """
    Executa a engenharia uma vez e retorna
    as transações indexadas por TransactionID.
    """
    result = apply_feature_engineering_spark(temporal_transactions)

    return {row["TransactionID"]: row.asDict() for row in result.collect()}


# ============================================================
# 3. VALIDAÇÃO DAS COLUNAS GERADAS
# ============================================================

def test_apply_feature_engineering_creates_expected_columns(temporal_transactions,):
    """Verifica a criação de todas as features esperadas."""

    result = apply_feature_engineering_spark(temporal_transactions)

    expected_new_cols = {
        "DeviceType_isnull",
        "TransactionAmt_log",
        "TransactionAmt_round",
        "transaction_hour",
        "transaction_day",
        "is_morning",
        "is_afternoon",
        "is_night",
        "is_weekend",
        "email_match",
        "P_email_free",
        "R_email_free",
        "card1_tx_count",
        "card1_amt_mean",
        "card1_amt_std",
        "TransactionAmt_dev_card",
        "TransactionAmt_zscore_card",
        "addr1_tx_count",
        "addr1_amt_mean",
        "device_mismatch",
    }

    missing = expected_new_cols - set(result.columns)

    assert not missing, (f"Features não geradas: {sorted(missing)}")


# ============================================================
# 4. FEATURES DE DISPOSITIVO
# ============================================================

def test_apply_feature_engineering_device_isnull(engineered_rows):
    """Verifica a identificação de dispositivos ausentes."""

    assert engineered_rows["TX001"]["DeviceType_isnull"] == 0
    assert engineered_rows["TX002"]["DeviceType_isnull"] == 1


# ============================================================
# 5. FEATURES DE EMAIL
# ============================================================

def test_apply_feature_engineering_email_match(engineered_rows):
    """Verifica igualdade entre domínios de email."""

    assert engineered_rows["TX001"]["email_match"] == 1
    assert engineered_rows["TX002"]["email_match"] == 0
    assert engineered_rows["TX003"]["email_match"] == 0

    assert engineered_rows["TX001"]["P_email_free"] == 1
    assert engineered_rows["TX002"]["R_email_free"] == 1


# ============================================================
# 6. AUSÊNCIA DE VAZAMENTO TEMPORAL
# ============================================================

def test_card_count_uses_only_past_transactions(engineered_rows):
    """
    A contagem histórica não pode incluir
    a transação atual nem transações futuras.
    """

    assert engineered_rows["TX001"]["card1_tx_count"] == 0
    assert engineered_rows["TX002"]["card1_tx_count"] == 1
    assert engineered_rows["TX003"]["card1_tx_count"] == 2

    # Primeiro registro de outro cartão.
    assert engineered_rows["TX004"]["card1_tx_count"] == 0


def test_card_mean_uses_only_past_transactions(engineered_rows):
    """
    Verifica as médias históricas do cartão 1234.

    TX001: sem histórico -> NULL
    TX002: média de [100] -> 100
    TX003: média de [100, 200] -> 150
    """

    assert engineered_rows["TX001"]["card1_amt_mean"] is None

    assert engineered_rows["TX002"]["card1_amt_mean"] == pytest.approx(100.0)
    assert engineered_rows["TX003"]["card1_amt_mean"] == pytest.approx(150.0)


def test_card_std_requires_sufficient_history(engineered_rows):
    """
    Desvio-padrão amostral requer pelo menos
    duas transações históricas.
    """

    assert engineered_rows["TX001"]["card1_amt_std"] is None
    assert engineered_rows["TX002"]["card1_amt_std"] is None

    assert engineered_rows["TX003"]["card1_amt_std"] == pytest.approx(70.710678,rel=1e-5)


def test_card_zscore_uses_historical_statistics(engineered_rows):
    """
    Para TX003:
      valor atual = 500
      média histórica = 150
      desvio histórico = sqrt(5000)

      z-score = (500 - 150) / sqrt(5000)
    """

    expected = 350.0 / (5000.0 ** 0.5)

    assert engineered_rows["TX003"]["TransactionAmt_zscore_card"] == pytest.approx(expected)


# ============================================================
# 7. AGREGAÇÕES HISTÓRICAS POR ENDEREÇO
# ============================================================

def test_address_aggregation_uses_only_past(engineered_rows):
    """Valida contagem e média históricas por endereço."""

    assert engineered_rows["TX001"]["addr1_tx_count"] == 0
    assert engineered_rows["TX002"]["addr1_tx_count"] == 1
    assert engineered_rows["TX003"]["addr1_tx_count"] == 2

    assert engineered_rows["TX003"]["addr1_amt_mean"] == pytest.approx(
        150.0
    )

    assert engineered_rows["TX004"]["addr1_tx_count"] == 0


# ============================================================
# 8. TRANSAÇÕES COM O MESMO TIMESTAMP
# ============================================================

def test_same_timestamp_does_not_leak(spark):
    """
    Transações no mesmo instante não devem
    utilizar umas às outras como histórico.
    """

    data = [
        ("TX001", 100, 100.0, 1234, 101, "mobile", "gmail.com", "gmail.com"),
        ("TX002", 100, 200.0, 1234, 101, "mobile", "gmail.com", "gmail.com"),
        ("TX003", 200, 300.0, 1234, 101, "mobile", "gmail.com", "gmail.com"),
    ]

    columns = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "DeviceType",
        "P_emaildomain",
        "R_emaildomain",
    ]

    df = spark.createDataFrame(data, columns)

    result = apply_feature_engineering_spark(df)

    rows = {
        row["TransactionID"]: row.asDict()
        for row in result.collect()
    }

    assert rows["TX001"]["card1_tx_count"] == 0
    assert rows["TX002"]["card1_tx_count"] == 0
    assert rows["TX003"]["card1_tx_count"] == 2

    assert rows["TX003"]["card1_amt_mean"] == pytest.approx(
        150.0
    )


# ============================================================
# 9. ISOLAMENTO ENTRE ENTIDADES
# ============================================================

def test_different_cards_have_independent_history(engineered_rows):
    """
    O histórico de um cartão não pode interferir
    nas agregações de outro cartão.
    """

    assert engineered_rows["TX003"]["card1_tx_count"] == 2
    assert engineered_rows["TX004"]["card1_tx_count"] == 0

    assert engineered_rows["TX004"]["card1_amt_mean"] is None


# ============================================================
# 10. VALIDAÇÃO DAS COLUNAS OBRIGATÓRIAS
# ============================================================

def test_missing_required_column_raises_error(temporal_transactions,):
    """Verifica falha explícita quando falta uma coluna."""

    df = temporal_transactions.drop("TransactionAmt")

    with pytest.raises(ValueError, match="TransactionAmt"):
        apply_feature_engineering_spark(df)
