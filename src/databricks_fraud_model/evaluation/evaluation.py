
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class EvaluationModelFraud:
    """
    Avaliação temporal de modelos de classificação binária.

    Estratégias:
        1. Walk-Forward Validation
        2. Out-of-Time Validation

    Métricas:
        Precision, Recall, F1, PR-AUC,
        ROC-AUC, Brier Score e KS.

    Requisitos:
        - Modelo compatível com a API do scikit-learn.
        - Implementação de fit() e predict_proba().
        - DataFrame Pandas ordenável temporalmente.
        - Target binário: 0 (legítima), 1 (fraude).
    """

    def __init__(self,model,target_col="target",time_col="TransactionDT",threshold=0.5,):
        self.model = model
        self.target_col = target_col
        self.time_col = time_col
        self.threshold = threshold

    # ========================================================
    # 1. CÁLCULO DAS MÉTRICAS
    # ========================================================

    def _calculate_metrics(self, y_true, y_proba):
        """
        Calcula métricas a partir das probabilidades
        previstas para a classe positiva.
        """

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        y_pred = (y_proba >= self.threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Average Precision resume a curva Precision-Recall.
        pr_auc = (average_precision_score(y_true, y_proba) if np.any(y_true == 1) else np.nan)

        # ROC-AUC e KS precisam de ambas as classes.
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_proba)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            ks = np.max(tpr - fpr)

        else:
            roc_auc = np.nan
            ks = np.nan

        brier = brier_score_loss(y_true, y_proba)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "ks": ks,
        }

    # ========================================================
    # 2. WALK-FORWARD VALIDATION
    # ========================================================

    def evaluation_walk_forward(self,df,feature_cols,n_splits=5,initial_train_size=0.5,window_type="expanding",train_window_size=None,gap=0,):
        """
        Avaliação temporal em múltiplos folds.

        window_type:
            expanding -> treinamento acumulativo
            sliding   -> janela de treinamento fixa

        gap:
            Quantidade de observações excluídas
            entre treino e validação.

        Observação:
            gap é medido em linhas, não em dias.
        """

        if window_type not in {"expanding", "sliding"}:
            raise ValueError("window_type deve ser 'expanding' ou 'sliding'.")

        if n_splits < 1:
            raise ValueError("n_splits deve ser >= 1.")

        if not 0 < initial_train_size < 1:
            raise ValueError("initial_train_size deve estar entre 0 e 1.")

        if gap < 0:
            raise ValueError("gap não pode ser negativo.")

        # Ordenação cronológica.
        df = (df.sort_values(self.time_col).reset_index(drop=True))

        n = len(df)
        initial_size = int(n * initial_train_size)

        # Reserva espaço para os folds futuros.
        remaining = n - initial_size
        fold_size = remaining // n_splits

        if fold_size <= gap:
            raise ValueError("Dados insuficientes para os folds e gap.")

        if window_type == "sliding":
            if train_window_size is None:
                train_window_size = initial_size

            if train_window_size <= 0:
                raise ValueError("train_window_size deve ser positivo.")

        results = []

        for fold in range(n_splits):

            train_end = initial_size + fold * fold_size
            val_start = train_end + gap

            val_end = (
                initial_size + (fold + 1) * fold_size
                if fold < n_splits - 1
                else n
            )

            if window_type == "expanding":
                train_start = 0
            else:
                train_start = max(0, train_end - train_window_size)

            train_df = df.iloc[train_start:train_end]
            val_df = df.iloc[val_start:val_end]

            if train_df.empty or val_df.empty:
                raise ValueError(f"Fold {fold + 1} possui conjunto vazio.")

            X_train = train_df[feature_cols]
            y_train = train_df[self.target_col]

            X_val = val_df[feature_cols]
            y_val = val_df[self.target_col]

            # Novo estimador em cada fold.
            model = clone(self.model)

            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = self._calculate_metrics(y_val, y_proba)

            results.append({
                "fold": fold + 1,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "train_end": train_df[self.time_col].max(),
                "val_start": val_df[self.time_col].min(),
                **metrics,
            })

        return pd.DataFrame(results)

    # ========================================================
    # 3. OUT-OF-TIME VALIDATION
    # ========================================================

    def evaluation_out_of_time(self,train_df,oot_df,feature_cols,):
        """
        Treina no período histórico e avalia
        em um período futuro independente.

        O conjunto OOT não deve ser utilizado
        para tuning ou escolha do threshold.
        """

        train_df = train_df.sort_values(self.time_col)

        oot_df = oot_df.sort_values(self.time_col)

        # Impede sobreposição temporal.
        if (train_df[self.time_col].max() >= oot_df[self.time_col].min()):
            raise ValueError(
                "O período OOT deve começar "
                "após o período de treinamento."
            )

        X_train = train_df[feature_cols]
        y_train = train_df[self.target_col]

        X_oot = oot_df[feature_cols]
        y_oot = oot_df[self.target_col]

        model = clone(self.model)

        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_oot)[:, 1]

        metrics = self._calculate_metrics(y_oot, y_proba)

        return pd.DataFrame([{
            "train_size": len(train_df),
            "oot_size": len(oot_df),
            "train_end": train_df[self.time_col].max(),
            "oot_start": oot_df[self.time_col].min(),
            **metrics,
        }])
