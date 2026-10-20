"""
Scoring via REST endpoint do modelo em producao.

Modulo responsavel por enviar dados ao serving endpoint do modelo
receber predicoes e persistir resultados em camadas Bronze (eventos brutos) e Gold (predicoes + features).

"""

# --- Imports padrao ---
import time                     # Medicao de latencia (perf_counter para alta precisao)
import uuid                     # Geracao de request_id unico por chamada
import logging                  # Logger para rastreabilidade de erros
from datetime import datetime, timezone  # Timestamps UTC para eventos

# --- Imports de terceiros ---
import pandas as pd             # Manipulacao de DataFrames para predicoes
from databricks.sdk import WorkspaceClient  # Cliente SDK para chamadas REST ao endpoint

# --- Imports do proprio pacote ---
from databricks_fraud_model.features import MODEL_FEATURES, FEATURE_CONTRACT_VERSION
from databricks_fraud_model.utils import load_config

# --- Configuracao global do modulo ---
# Carrega o config uma vez na importacao (config/project_config.yml resolve paths relativos)
_cfg = load_config()
ENDPOINT_NAME = "model_fraud"                                      # Nome do serving endpoint no Databricks
BRONZE_TABLE = f"{_cfg['catalog']}.bronze.monitoring_raw_events"  # Tabela de eventos brutos (append-only)
GOLD_TABLE = f"{_cfg['catalog']}.gold.model_predictions"           # Tabela de predicoes (upsert via MERGE)

# Logger do modulo — permite rastrear requests bem-sucedidas e falhas
logger = logging.getLogger(__name__)


def _clean_records(df, features, n=None):
    """
    Prepara os dados para envio ao endpoint REST.

    Seleciona apenas as colunas de features do modelo, limita o numero de
    registros (se n for fornecido) e converte NaN/NaT para None, pois o
    formato JSON nao suporta valores NaN nativamente.

    Args:
        df:       DataFrame pandas com os dados de producao.
        features: lista de colunas a enviar (subset de MODEL_FEATURES).
        n:        numero maximo de registros (None = todos).

    Returns:
        Lista de dicts, um por registro, com NaN substituido por None.
    """
    # Seleciona as colunas de features e limita o numero de linhas
    sample = df[features].head(n) if n else df[features]
    # Converte para lista de dicionarios (formato esperado pelo endpoint REST)
    records = sample.to_dict(orient='records')
    # Substitui NaN por None para conformidade com JSON
    return [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in records]


def score_batch(df, features=None, n=None, endpoint_name=ENDPOINT_NAME):
    """
    Envia features diretamente no payload REST.

    Args:
        df: DataFrame pandas com os dados de producao (deve conter as features).
        features: lista de colunas a enviar. Default: MODEL_FEATURES (152).
        n: numero de registros a enviar. Default: todos.
        endpoint_name: nome do serving endpoint.

    Returns:
        DataFrame com TransactionID, probability_fraud, is_fraud, latency_ms,
        endpoint_name, n_records, request_id, status.
    """
    if features is None:
        features = MODEL_FEATURES

    # Gera um ID unico para rastrear esta requisicao em logs e eventos
    request_id = str(uuid.uuid4())

    # Cria o cliente do Databricks SDK (auto-autenticado com as credenciais do usuario)
    w = WorkspaceClient()

    # Prepara o payload: seleciona features, limpa NaN -> None
    clean = _clean_records(df, features, n)
    payload = {'dataframe_records': clean}

    # --- Chamada ao endpoint REST com medicao de latencia ---
    t0 = time.perf_counter()
    try:
        # POST /serving-endpoints/{endpoint}/invocations com formato dataframe_records
        response = w.api_client.do(
            'POST',
            f'/serving-endpoints/{endpoint_name}/invocations',
            body=payload,
            headers={'Content-Type': 'application/json'},
        )

        # Latencia em milissegundos (perf_counter = alta resolucao)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        status = 'success'
        logger.info(f"Scoring request {request_id} succeeded: {len(clean)} records, {latency_ms}ms")
        
    except Exception as e:
        # Em caso de erro, ainda medimos a latencia para diagnostico
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        status = 'error'
        logger.error(f"Scoring request {request_id} failed after {latency_ms}ms: {e}")
        raise  # Re-lanca a excecao para que o chamador decida como tratar

    # --- Monta o DataFrame de predicoes com metadados ---
    preds = pd.DataFrame(response['predictions'])
    preds['latency_ms'] = latency_ms        # Latencia da chamada HTTP (uma por batch)
    preds['endpoint_name'] = endpoint_name    # Nome do endpoint para rastreabilidade
    preds['n_records'] = len(preds)          # Numero de predicoes retornadas
    preds['request_id'] = request_id          # UUID unico da requisicao
    preds['status'] = status                  # 'success' ou 'error'

    # Anexa TransactionID do DataFrame original para permitir JOIN com dados de producao
    if 'TransactionID' in df.columns:
        ids = df['TransactionID'].head(n) if n else df['TransactionID']
        preds['TransactionID'] = ids.values

    return preds


def score_single(record, features=None, endpoint_name=ENDPOINT_NAME):
    """
    Scoring de um unico registro via endpoint REST.

    Util para predicoes em tempo real (uma transacao por vez) ou
    para testes rapidos do endpoint sem enviar um batch completo.

    Args:
        record:        dict com as features do modelo (chave = nome da feature).
        features:      lista de colunas a enviar. Default: MODEL_FEATURES (152).
        endpoint_name: nome do serving endpoint.

    Returns:
        dict com probability_fraud, is_fraud e latency_ms.
    """
    if features is None:
        features = MODEL_FEATURES

    w = WorkspaceClient()
    # Filtra apenas as features do modelo e limpa NaN -> None
    clean = {k: (None if pd.isna(v) else v) for k, v in record.items() if k in features}

    # Chamada REST com medicao de latencia
    t0 = time.perf_counter()
    response = w.api_client.do(
        'POST',
        f'/serving-endpoints/{endpoint_name}/invocations',
        body={'dataframe_records': [clean]},  # Endpoint espera uma lista de records
        headers={'Content-Type': 'application/json'},
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Extrai a primeira (e unica) predicao do response
    result = response['predictions'][0]
    result['latency_ms'] = latency_ms
    return result


def save_to_bronze(preds_df, table_name=None, catalog=None):
    """
    Salva eventos brutos de scoring na camada Bronze para observabilidade.

    Cada chamada ao endpoint e registrada como um evento individual, contendo
    a predicao, latencia e metadados de rastreabilidade. Esta tabela e
    append-only (cada requisicao e um evento historico distinto) e e a fonte
    para o pipeline ETL de monitoring (Bronze -> Silver -> Gold).

    Colunas registradas:
        event_timestamp          — UTC timestamp da gravacao
        TransactionID            — ID da transacao (pode ser None em testes)
        probability_fraud       — probabilidade retornada pelo modelo
        is_fraud                — classificacao binaria (0/1) apos threshold
        latency_ms              — latencia da chamada HTTP (ms)
        endpoint_name           — nome do serving endpoint
        n_records               — numero de predicoes no batch
        status                  — 'success' ou 'error'
        request_id              — UUID unico da requisicao
        feature_contract_version — versao do contrato de features usado

    Args:
        preds_df:   DataFrame retornado por score_batch().
        table_name: nome completo da tabela Bronze (catalog.schema.table).
        catalog:    catalogo UC para override do config (opcional).
    """
    # Importa Spark aqui (lazy) para nao falhar em contextos sem Spark (ex: testes unitarios)
    from pyspark.sql import SparkSession

    # Resolve o nome da tabela: parametro > catalog override > config default
    if table_name is None:
        _cat = catalog or _cfg['catalog']
        table_name = f"{_cat}.bronze.monitoring_raw_events"

    spark = SparkSession.builder.getOrCreate()

    # Garante que o schema (catalog.bronze) existe no Unity Catalog
    schema_name = '.'.join(table_name.split('.')[:2])
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    # --- Monta o DataFrame de eventos com todos os metadados ---
    n = len(preds_df)
    events = pd.DataFrame({
        'event_timestamp': [datetime.now(timezone.utc)] * n,  # UTC timestamp
        'TransactionID': preds_df['TransactionID'].values if 'TransactionID' in preds_df.columns else [None] * n,
        'probability_fraud': preds_df['probability_fraud'].values,
        'is_fraud': preds_df['is_fraud'].values,
        'latency_ms': preds_df['latency_ms'].values,
        'endpoint_name': preds_df['endpoint_name'].values if 'endpoint_name' in preds_df.columns else [ENDPOINT_NAME] * n,
        'n_records': preds_df['n_records'].values if 'n_records' in preds_df.columns else [n] * n,
        'status': preds_df['status'].values if 'status' in preds_df.columns else ['success'] * n,
        'request_id': preds_df['request_id'].values if 'request_id' in preds_df.columns else [str(uuid.uuid4())] * n,
        'feature_contract_version': [FEATURE_CONTRACT_VERSION] * n,  # Rastreia qual contrato gerou as features
    })

    # Converte para Spark DataFrame e salva em modo append (cada evento e historico)
    spark_df = spark.createDataFrame(events)
    spark_df.write.mode("append").saveAsTable(table_name)
    return events


def save_predictions(df, preds_df, features=None, table_name=None, catalog=None):
    """
    Salva predicoes + features usadas em tabela Gold para monitoring.

    Usa MERGE (upsert) por TransactionID para garantir idempotencia:
    re-processar a mesma transacao atualiza a predicao em vez de duplicar.
    Isso evita que re-runs do job de inference criem linhas duplicadas na Gold.

    Fluxo:
        1. Salva eventos brutos em Bronze (append-only, para observabilidade)
        2. Monta DataFrame com features + predicoes + metadados
        3. Se a tabela Gold existe: MERGE por TransactionID (upsert)
        4. Se nao existe: cria a tabela (errorifexists para evitar overwrite acidental)

    Args:
        df:         DataFrame original com os dados de producao (contem as features).
        preds_df:   DataFrame retornado por score_batch().
        features:   lista de features a salvar. Default: MODEL_FEATURES (152).
        table_name: nome da tabela Gold (catalog.schema.table).
        catalog:    catalogo UC para override do config (opcional).

    Returns:
        DataFrame pandas com as features + predicoes salvas.
    """
    # Importa Spark e DeltaTable aqui (lazy) para compatibilidade com testes
    from pyspark.sql import SparkSession
    from delta.tables import DeltaTable

    # --- Resolve parametros com fallback para config ---
    if features is None:
        features = MODEL_FEATURES
    if table_name is None:
        _cat = catalog or _cfg['catalog']
        table_name = f"{_cat}.gold.model_predictions"

    spark = SparkSession.builder.getOrCreate()

    # Passo 1: Salva eventos brutos em Bronze (rastreabilidade de cada requisicao)
    save_to_bronze(preds_df, catalog=catalog)

    # --- Passo 2: Monta o DataFrame resultado com features + predicoes ---
    n = len(preds_df)
    result = df[features].head(n).copy()  # Seleciona apenas as features do modelo
    result['TransactionID'] = preds_df['TransactionID'].values if 'TransactionID' in preds_df.columns else df['TransactionID'].head(n).values
    result['probability_fraud'] = preds_df['probability_fraud'].values
    result['is_fraud'] = preds_df['is_fraud'].values
    result['prediction_timestamp'] = datetime.now(timezone.utc)  # UTC timestamp da predicao
    result['request_id'] = preds_df['request_id'].values if 'request_id' in preds_df.columns else [str(uuid.uuid4())] * n
    result['feature_contract_version'] = FEATURE_CONTRACT_VERSION  # Versao do contrato usado

    # Garante que o schema (catalog.gold) existe no Unity Catalog
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {'.'.join(table_name.split('.')[:2])}")
    spark_df = spark.createDataFrame(result)

    # --- Passo 3: Idempotencia via MERGE por TransactionID ---
    # Se a tabela ja existe: upsert (update + insert) evita duplicatas
    if spark.catalog.tableExists(table_name):
        delta_table = DeltaTable.forName(spark, table_name)
        (
            delta_table.alias("dest")
            .merge(
                spark_df.alias("src"),
                "dest.TransactionID = src.TransactionID"  # Chave de merge: TransactionID unica
            )
            .whenMatchedUpdateAll()    # Atualiza se TransactionID ja existe
            .whenNotMatchedInsertAll()  # Insere se TransactionID e nova
            .execute()
        )
    else:
        # Primeira criacao da tabela: errorifexists protege contra overwrite
        spark_df.write.mode("errorifexists").saveAsTable(table_name)

    return result
