from pyspark.sql import Window
from pyspark.sql import functions as F

from .features import FREE_DOMAINS


def apply_feature_engineering_spark(df):
    """
    Aplica engenharia de atributos em PySpark.

    As agregações históricas consideram somente transações
    anteriores ao TransactionDT da transação atual.

    O DataFrame precisa conter o histórico necessário
    para calcular corretamente essas agregações.
    """

    # ========================================================
    # 1. VALIDAÇÃO DAS COLUNAS
    # ========================================================

    required_columns = {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "addr1",
        "DeviceType",
        "P_emaildomain",
        "R_emaildomain",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colunas ausentes: {sorted(missing_columns)}")

    # TransactionDT representa segundos relativos no IEEE-CIS.
    df = df.withColumn("TransactionDT",F.col("TransactionDT").cast("long"),)

    # ========================================================
    # 2. FEATURES DE DISPOSITIVO
    # ========================================================

    df = df.withColumn("DeviceType_isnull",F.when(F.col("DeviceType").isNull(), 1).otherwise(0),)

    # ========================================================
    # 3. FEATURES DE TRANSAÇÃO
    # ========================================================

    df = df.withColumn(
        "TransactionAmt_log",
        F.when(
            F.col("TransactionAmt") >= 0,
            F.log1p(F.col("TransactionAmt")),
        ),
    )

    df = df.withColumn(
        "TransactionAmt_round",
        F.when(
            F.col("TransactionAmt") % 1 == 0,
            1,
        ).otherwise(0),
    )

    # ========================================================
    # 4. FEATURES TEMPORAIS
    # ========================================================

    df = df.withColumn("transaction_hour",((F.col("TransactionDT") / 3600) % 24).cast("int"),)

    df = df.withColumn("transaction_day",((F.col("TransactionDT") / 86400) % 7).cast("int"),)

    df = df.withColumn(
        "is_morning",
        F.when(
            (F.col("transaction_hour") >= 6)
            & (F.col("transaction_hour") < 12),
            1,
        ).otherwise(0),
    )

    df = df.withColumn(
        "is_afternoon",
        F.when(
            (F.col("transaction_hour") >= 12)
            & (F.col("transaction_hour") < 18),
            1,
        ).otherwise(0),
    )

    df = df.withColumn(
        "is_night",
        F.when(
            (F.col("transaction_hour") >= 18)
            | (F.col("transaction_hour") < 6),
            1,
        ).otherwise(0),
    )

    # Indicador relativo ao ciclo de sete dias.
    df = df.withColumn(
        "is_weekend",
        F.when(
            F.col("transaction_day") >= 5,
            1,
        ).otherwise(0),
    )

    # ========================================================
    # 5. FEATURES DE EMAIL
    # ========================================================

    df = df.withColumn(
        "email_match",
        F.when(
            F.col("P_emaildomain").isNotNull()
            & F.col("R_emaildomain").isNotNull()
            & (
                F.col("P_emaildomain")
                == F.col("R_emaildomain")
            ),
            1,
        ).otherwise(0),
    )

    df = df.withColumn(
        "P_email_free",
        F.when(
            F.col("P_emaildomain").isin(*FREE_DOMAINS),
            1,
        ).otherwise(0),
    )

    df = df.withColumn(
        "R_email_free",
        F.when(
            F.col("R_emaildomain").isin(*FREE_DOMAINS),
            1,
        ).otherwise(0),
    )

    # ========================================================
    # 6. JANELAS HISTÓRICAS
    # ========================================================

    # rangeBetween(..., -1) exclui transações do mesmo
    # instante e todas as transações futuras.
    w_card1 = (
        Window.partitionBy("card1")
        .orderBy("TransactionDT")
        .rangeBetween(Window.unboundedPreceding, -1)
    )

    w_addr1 = (
        Window.partitionBy("addr1")
        .orderBy("TransactionDT")
        .rangeBetween(Window.unboundedPreceding, -1)
    )

    # ========================================================
    # 7. AGREGAÇÕES HISTÓRICAS POR CARTÃO
    # ========================================================

    df = df.withColumn("card1_tx_count",F.count("TransactionID").over(w_card1),)
    df = df.withColumn("card1_amt_mean",F.avg("TransactionAmt").over(w_card1),)
    df = df.withColumn("card1_amt_std",F.stddev_samp("TransactionAmt").over(w_card1),)
    df = df.withColumn("TransactionAmt_dev_card",F.col("TransactionAmt") - F.col("card1_amt_mean"),)

    df = df.withColumn(
        "TransactionAmt_zscore_card",
        F.when(
            F.col("card1_amt_std") > 0,
            F.col("TransactionAmt_dev_card")
            / F.col("card1_amt_std"),
        ),
    )

    # ========================================================
    # 8. AGREGAÇÕES HISTÓRICAS POR ENDEREÇO
    # ========================================================

    df = df.withColumn("addr1_tx_count",F.count("TransactionID").over(w_addr1),)
    df = df.withColumn("addr1_amt_mean",F.avg("TransactionAmt").over(w_addr1),)

    # ========================================================
    # 9. COMPATIBILIDADE COM O MODELO
    # ========================================================

    # Mantido como placeholder do schema existente.
    # Não representa uma detecção real de divergência.
    df = df.withColumn("device_mismatch",F.lit(0),)

    return df
