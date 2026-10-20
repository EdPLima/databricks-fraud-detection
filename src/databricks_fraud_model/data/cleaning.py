"""
Limpeza e padronizacao de dados para a camada Silver.

Funcoes reutilizaveis extraidas do notebook pipeline/silver/02_silver_clean.
Cada funcao e testavel isoladamente e pode ser chamada por notebooks,
pipelines ou outros modulos sem depender do contexto do notebook.

Fluxo tipico (no notebook pipeline/silver/02_silver_clean):
    from databricks_fraud_model.data import clean_silver_data

    transactions = spark.table(f"{catalog}.bronze.{batch}_transaction_raw")
    identity = spark.table(f"{catalog}.bronze.{batch}_identity_raw")
    dados_silver = clean_silver_data(transactions, identity)
    # ... persistencia (MERGE ou overwrite) fica no notebook
"""

from pyspark.sql import Window
from pyspark.sql import functions as F

# ============================================================
# 1. DEDUPLICACAO DE IDENTIDADE
# ============================================================

def deduplicate_identity(identity_df):
    """
    Remove duplicatas de identidade por TransactionID.

    Se houver `ingestion_timestamp`, mantem o registro mais recente.
    Caso contrario, remove apenas duplicatas identicas e valida que
    nao ha conflitos para o mesmo TransactionID.

    Args:
        identity_df: Spark DataFrame da tabela bronze de identidade.

    Returns:
        Spark DataFrame deduplicado por TransactionID.

    Raises:
        ValueError: se houver identidades conflitantes sem timestamp.
    """
    if "ingestion_timestamp" in identity_df.columns:
        # Mantem o registro mais recente por TransactionID
        w = (
            Window
            .partitionBy("TransactionID")
            .orderBy(F.col("ingestion_timestamp").desc_nulls_last())
        )
        return (
            identity_df
            .withColumn("_row_number", F.row_number().over(w))
            .filter(F.col("_row_number") == 1)
            .drop("_row_number")
        )
    else:
        # Sem timestamp: remove apenas duplicatas identicas
        identity_df = identity_df.dropDuplicates()

        # Verifica se ainda ha TransactionIDs duplicados com dados diferentes
        conflicts = (
            identity_df
            .groupBy("TransactionID")
            .count()
            .filter(F.col("count") > 1)
            .limit(1)
            .count()
        )
        if conflicts:
            raise ValueError(
                "Existem identidades conflitantes para "
                "o mesmo TransactionID sem timestamp para desempate."
            )
        return identity_df


# ============================================================
# 2. JOIN TRANSACTIONS + IDENTITY
# ============================================================

def join_transactions_identity(transactions_df, identity_df):
    """
    LEFT JOIN entre transacoes e identidade por TransactionID.

    Exclui colunas tecnicas da Bronze (ingestion_timestamp, dataset_type)
    para evitar duplicacao no resultado.

    Args:
        transactions_df: Spark DataFrame de transacoes (Bronze).
        identity_df:      Spark DataFrame de identidade ja deduplicado.

    Returns:
        Spark DataFrame com transacoes + colunas de identidade.
    """
    # Seleciona apenas colunas de identidade (excluindo tecnicas)
    identity_columns = [
        col for col in identity_df.columns
        if col not in {"TransactionID", "ingestion_timestamp", "dataset_type"}
    ]

    return transactions_df.join(
        identity_df.select("TransactionID", *identity_columns),
        on="TransactionID",
        how="left",
    )


# ============================================================
# 3. PADRONIZACAO DE NOMES DE COLUNAS
# ============================================================

def standardize_column_names(df):
    """
    Padroniza nomes de colunas do DataFrame.

    - Renomeia colunas `id-XX` para `id_XX` (hifen → underline).
    - Renomeia `isFraud` para `target` (quando existir).

    Args:
        df: Spark DataFrame com nomes de colunas nao padronizados.

    Returns:
        Spark DataFrame com nomes de colunas padronizados.
    """
    # id-XX → id_XX
    for col_name in df.columns:
        if col_name.startswith("id-"):
            df = df.withColumnRenamed(col_name, col_name.replace("-", "_"))

    # isFraud → target (convencao do projeto)
    if "isFraud" in df.columns:
        df = df.withColumnRenamed("isFraud", "target")

    return df


# ============================================================
# 4. TRATAMENTO DE id_34
# ============================================================

def parse_id_34(df):
    """
    Extrai o valor numerico da variavel id_34.

    id_34 tem formato textual como ":5" ou ":12" — esta funcao
    extrai o numero apos os dois pontos e converte para float.

    Args:
        df: Spark DataFrame que pode conter a coluna id_34.

    Returns:
        Spark DataFrame com id_34 convertida para float (ou None).
    """
    if "id_34" not in df.columns:
        return df

    # Extrai o numero apos os dois pontos
    extracted = F.regexp_extract(F.col("id_34"), r":(\d+)", 1)

    return df.withColumn(
        "id_34",
        F.when(extracted != "", extracted.cast("float")).otherwise(
            F.lit(None).cast("float")
        ),
    )


# ============================================================
# 5. VALIDACAO DE CHAVES
# ============================================================

def validate_transaction_ids(df):
    """
    Valida e limpa chaves de TransactionID.

    - Remove registros sem TransactionID.
    - Remove duplicatas identicas.
    - Verifica unicidade por TransactionID.

    Args:
        df: Spark DataFrame com a coluna TransactionID.

    Returns:
        Spark DataFrame limpo com TransactionIDs unicos.

    Raises:
        ValueError: se houver multiplos registros para o mesmo TransactionID.
    """
    # Remove registros sem identificador
    df = df.filter(F.col("TransactionID").isNotNull())

    # Remove apenas duplicatas inteiramente identicas
    df = df.dropDuplicates()

    # Verifica unicidade por TransactionID
    duplicates = (
        df.groupBy("TransactionID")
        .count()
        .filter(F.col("count") > 1)
        .limit(1)
        .count()
    )
    if duplicates:
        raise ValueError(
            "Existem multiplos registros conflitantes para o mesmo TransactionID."
        )

    return df


# ============================================================
# 6. ORQUESTRADOR: CLEAN_SILVER_DATA
# ============================================================

def clean_silver_data(transactions_df, identity_df):
    """
    Pipeline completo de limpeza Silver: dedup, join, padronizacao, validacao.

    Esta e a funcao principal que o notebook pipeline/silver/02_silver_clean
    chama. O notebook fica responsavel apenas por ler as tabelas Bronze,
    chamar esta funcao e persistir o resultado (MERGE ou overwrite).

    Etapas executadas (em ordem):
        1. Deduplica identidade por TransactionID
        2. LEFT JOIN transacoes + identidade
        3. Padroniza nomes de colunas (id-XX → id_XX, isFraud → target)
        4. Extrai valor numerico de id_34
        5. Valida unicidade de TransactionID

    Args:
        transactions_df: Spark DataFrame da tabela bronze de transacoes.
        identity_df:      Spark DataFrame da tabela bronze de identidade.

    Returns:
        Spark DataFrame limpo e padronizado, pronto para persistir em Silver.

    Example:
        >>> from databricks_fraud_model.data import clean_silver_data
        >>> transactions = spark.table(f"{catalog}.bronze.{batch}_transaction_raw")
        >>> identity = spark.table(f"{catalog}.bronze.{batch}_identity_raw")
        >>> dados_silver = clean_silver_data(transactions, identity)
    """
    # Passo 1: Deduplica identidade
    identity_df = deduplicate_identity(identity_df)

    # Passo 2: JOIN transacoes + identidade
    df = join_transactions_identity(transactions_df, identity_df)

    # Passo 3: Padroniza nomes de colunas
    df = standardize_column_names(df)

    # Passo 4: Trata id_34
    df = parse_id_34(df)

    # Passo 5: Valida chaves
    df = validate_transaction_ids(df)

    return df
