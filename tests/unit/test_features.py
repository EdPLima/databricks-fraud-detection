from databricks_fraud_model.features import (
    MODEL_FEATURES,
    FEATURES_DROP,
    MODEL_KEEP,
    REMOVED_QUALITY,
    REMOVED_REDUNDANCY,
    FREE_DOMAINS,
    apply_feature_engineering_spark,
)


def test_model_features_count():
    assert len(MODEL_FEATURES) == 152


def test_model_features_no_duplicates():
    assert len(MODEL_FEATURES) == len(set(MODEL_FEATURES))


def test_model_keep_not_in_features_drop():
    for feature in MODEL_KEEP:
        assert feature not in FEATURES_DROP, f"MODEL_KEEP feature '{feature}' should not be in FEATURES_DROP"


def test_removed_lists_no_overlap_with_model_keep():
    for feature in MODEL_KEEP:
        assert feature not in REMOVED_QUALITY or feature in MODEL_KEEP
        assert feature not in REMOVED_REDUNDANCY or feature in MODEL_KEEP


def test_free_domains_is_list():
    assert isinstance(FREE_DOMAINS, list)
    assert "gmail.com" in FREE_DOMAINS
    assert len(FREE_DOMAINS) > 0


def test_apply_feature_engineering_creates_expected_columns(spark, sample_df):
    sdf = spark.createDataFrame(sample_df.drop(columns=["dist1", "card2", "card3", "card5"]))
    result = apply_feature_engineering_spark(sdf)

    expected_new_cols = [
    "DeviceType_isnull", "TransactionAmt_log", "TransactionAmt_round",
    "transaction_hour", "transaction_day", "is_morning", "is_afternoon",
    "is_night", "is_weekend", "email_match", "P_email_free", "R_email_free",
    "card1_tx_count", "card1_amt_mean", "card1_amt_std",
    "TransactionAmt_dev_card", "TransactionAmt_zscore_card",
    "addr1_tx_count", "addr1_amt_mean", "device_mismatch",
    ]
    result_cols = set(result.columns)
    for col in expected_new_cols:
        assert col in result_cols, f"Feature engineering should create column: {col}"


def test_apply_feature_engineering_device_isnull(spark, sample_df):
    sdf = spark.createDataFrame(sample_df.drop(columns=["dist1", "card2", "card3", "card5"]))
    result = apply_feature_engineering_spark(sdf).orderBy("TransactionID").toPandas()

    assert result.loc[0, "DeviceType_isnull"] == 0
    assert result.loc[1, "DeviceType_isnull"] == 1


def test_apply_feature_engineering_email_match(spark, sample_df):
    sdf = spark.createDataFrame(sample_df.drop(columns=["dist1", "card2", "card3", "card5"]))
    result = apply_feature_engineering_spark(sdf).orderBy("TransactionID").toPandas()

    assert result.loc[0, "email_match"] == 1
    assert result.loc[2, "email_match"] == 0
    assert result.loc[0, "P_email_free"] == 1
    assert result.loc[2, "P_email_free"] == 1
