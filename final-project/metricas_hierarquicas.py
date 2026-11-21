"""
Módulo de avaliação hierárquica (Ordem -> Família) para modelos Ultralytics YOLO-cls,
otimizado para baixo consumo de memória primária (RAM/VRAM) e datasets desbalanceados.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


# ==============================================================================
# Exceções Customizadas
# ==============================================================================


class EvaluationError(Exception):
    """Exceção base para falhas no pipeline de avaliação."""


class DegeneratePredictionError(EvaluationError):
    """Lançada quando o modelo produz probabilidades NaN, Inf ou nulas."""


class LabelAlignmentError(EvaluationError):
    """Lançada quando há inconsistência entre diretórios e model.names."""


# ==============================================================================
# Contratos de Dados
# ==============================================================================


@dataclass(frozen=True)
class ClassMetrics:
    """Métricas granulares de desempenho para uma única classe."""

    class_id: int
    class_name: str
    support: int
    predicted_count: int
    precision: float
    recall: float
    f1_score: float
    average_precision: float
    roc_auc: float


@dataclass(frozen=True)
class EvaluationReport:
    """Relatório consolidado para um nível taxonômico (Ordem ou Família)."""

    level_name: str
    num_samples: int
    num_classes: int
    confidence_threshold: float
    rejected_count: int
    top1_accuracy: float
    top5_accuracy: float
    log_loss_value: float
    map_macro: float
    map_weighted: float
    precision_macro: float
    precision_weighted: float
    precision_micro: float
    recall_macro: float
    recall_weighted: float
    recall_micro: float
    f1_macro: float
    f1_weighted: float
    f1_micro: float
    roc_auc_macro: float
    roc_auc_weighted: float
    confusion_matrix: NDArray[np.int64]
    confusion_matrix_normalized: NDArray[np.float64]
    per_class: dict[str, ClassMetrics]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confusion_matrix"] = self.confusion_matrix.tolist()
        payload["confusion_matrix_normalized"] = np.round(
            self.confusion_matrix_normalized, 6
        ).tolist()
        return payload

    def format_summary(self) -> str:
        lines = [
            "=" * 94,
            f"RELATÓRIO [{self.level_name.upper()}] | Amostras: {self.num_samples} | Classes: {self.num_classes}",
            f"Threshold de Confiança: {self.confidence_threshold:.2f} | Rejeitadas: {self.rejected_count}",
            "=" * 94,
            f"Top-1 Accuracy : {self.top1_accuracy:.4f} | Top-5 Accuracy : {self.top5_accuracy:.4f}",
            f"mAP (Macro)    : {self.map_macro:.4f} | mAP (Weighted) : {self.map_weighted:.4f}",
            f"F1 (Macro)     : {self.f1_macro:.4f} | F1 (Weighted)  : {self.f1_weighted:.4f} | F1 (Micro): {self.f1_micro:.4f}",
            f"Precision Macro: {self.precision_macro:.4f} | Recall Macro   : {self.recall_macro:.4f}",
            f"ROC-AUC (Macro): {self.roc_auc_macro:.4f} | Log-Loss       : {self.log_loss_value:.4f}",
            "-" * 94,
            f"{'ID':<4} {'Classe':<30} {'Supp':>6} {'Pred':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7} {'AUC':>7}",
            "-" * 94,
        ]
        for cm in self.per_class.values():
            ap_str = f"{cm.average_precision:.4f}" if not np.isnan(cm.average_precision) else "   N/A"
            auc_str = f"{cm.roc_auc:.4f}" if not np.isnan(cm.roc_auc) else "   N/A"
            lines.append(
                f"{cm.class_id:<4} {cm.class_name[:30]:<30} {cm.support:>6} {cm.predicted_count:>6} "
                f"{cm.precision:>7.4f} {cm.recall:>7.4f} {cm.f1_score:>7.4f} {ap_str:>7} {auc_str:>7}"
            )
        if self.warnings:
            lines.append("-" * 94)
            lines.append("ALERTAS DE VALIDAÇÃO:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        lines.append("=" * 94)
        return "\n".join(lines)


@dataclass(frozen=True)
class HierarchicalEvaluationReport:
    """Relatório completo da arquitetura hierárquica (Camada 1 + Camada 2)."""

    top_k_routing: int
    order_report: EvaluationReport
    family_report: EvaluationReport
    family_accuracy_given_correct_order: float
    correct_order_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k_routing": self.top_k_routing,
            "correct_order_count": self.correct_order_count,
            "family_accuracy_given_correct_order": round(
                self.family_accuracy_given_correct_order, 6
            ),
            "order_report": self.order_report.to_dict(),
            "family_report": self.family_report.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def format_summary(self) -> str:
        header = [
            "#" * 94,
            f"AVALIAÇÃO HIERÁRQUICA (Top-K Routing = {self.top_k_routing})",
            f"Acurácia Camada 1 (Ordem)                   : {self.order_report.top1_accuracy:.4f} "
            f"({self.correct_order_count}/{self.order_report.num_samples})",
            f"Acurácia Camada 2 | Ordem Correta (Cond.)   : {self.family_accuracy_given_correct_order:.4f}",
            f"Acurácia Ponta-a-Ponta (Família Global)     : {self.family_report.top1_accuracy:.4f}",
            "#" * 94,
            "",
        ]
        return (
            "\n".join(header)
            + self.order_report.format_summary()
            + "\n\n"
            + self.family_report.format_summary()
        )


# ==============================================================================
# Utilitário de Liberação de Memória
# ==============================================================================


def _release_memory() -> None:
    """Força coleta de lixo no Python e libera cache ocioso da GPU."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Motor Matemático (Zero-Copy / Processamento em Blocos)
# ==============================================================================


class ClassificationMetricsEvaluator:
    """
    Motor determinístico para cálculo de métricas multiclasse em blocos,
    evitando duplicação da matriz de probabilidades na RAM.
    """

    def __init__(
        self,
        class_names: Mapping[int, str],
        level_name: str = "classification",
        eps: float = 1e-7,
        strict_nan: bool = True,
        eval_chunk_size: int = 4096,
        allow_sub_stochastic: bool = False,
    ) -> None:
        if not class_names:
            raise ValueError("class_names não pode ser vazio.")
        self.id_to_name: dict[int, str] = {int(k): str(v) for k, v in class_names.items()}
        self.num_classes: int = len(self.id_to_name)
        self.all_labels: list[int] = list(range(self.num_classes))
        if set(self.id_to_name.keys()) != set(self.all_labels):
            raise LabelAlignmentError("As chaves de class_names devem ser 0..C-1.")
        self.level_name = level_name
        self.eps = eps
        self.strict_nan = strict_nan
        self.eval_chunk_size = eval_chunk_size
        self.allow_sub_stochastic = allow_sub_stochastic

    def _validate_and_sanitize_inplace(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], list[str]]:
        warnings_list: list[str] = []

        y_true_arr = np.asarray(y_true, dtype=np.int64)
        if isinstance(y_prob, np.ndarray) and y_prob.dtype == np.float32 and y_prob.flags.writeable:
            y_prob_arr = y_prob
        else:
            y_prob_arr = np.array(y_prob, dtype=np.float32, copy=True)

        if y_true_arr.ndim != 1 or y_prob_arr.ndim != 2:
            raise ValueError("Shapes inválidos: y_true deve ser 1D e y_prob 2D.")
        if y_true_arr.shape[0] == 0:
            raise ValueError("O conjunto de avaliação está vazio.")
        if y_true_arr.shape[0] != y_prob_arr.shape[0]:
            raise ValueError("Incompatibilidade no número de amostras entre y_true e y_prob.")
        if y_prob_arr.shape[1] != self.num_classes:
            raise ValueError(
                f"Incompatibilidade de classes: esperado {self.num_classes}, "
                f"recebido {y_prob_arr.shape[1]}."
            )
        if np.any((y_true_arr < 0) | (y_true_arr >= self.num_classes)):
            raise LabelAlignmentError(
                f"y_true contém índices fora de [0, {self.num_classes - 1}]."
            )

        num_samples = y_true_arr.shape[0]
        total_non_finite = 0
        total_zero_sum = 0
        max_drift = 0.0

        for start in range(0, num_samples, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, num_samples)
            block = y_prob_arr[start:end]

            non_finite_rows = ~np.all(np.isfinite(block), axis=1)
            if np.any(non_finite_rows):
                bad_count = int(np.sum(non_finite_rows))
                if self.strict_nan:
                    raise DegeneratePredictionError(
                        f"Probabilidades NaN/Inf detectadas no nível '{self.level_name}'."
                    )
                total_non_finite += bad_count
                block[non_finite_rows] = np.float32(1.0 / self.num_classes)

            if np.any(block < -self.eps):
                raise DegeneratePredictionError(
                    "Probabilidades negativas detectadas (saída sem Softmax)."
                )
            np.clip(block, 0.0, None, out=block)

            row_sums = np.sum(block, axis=1, keepdims=True, dtype=np.float64)
            zero_rows = (row_sums <= self.eps).ravel()
            if np.any(zero_rows):
                bad_zero = int(np.sum(zero_rows))
                if self.strict_nan:
                    raise DegeneratePredictionError(
                        f"Amostras com soma de probabilidades nula no nível '{self.level_name}'."
                    )
                total_zero_sum += bad_zero
                block[zero_rows] = np.float32(1.0 / self.num_classes)
                row_sums = np.sum(block, axis=1, keepdims=True, dtype=np.float64)

            if self.allow_sub_stochastic:
                # Em roteamento Top-K < C, sum(p) <= 1.0 é esperado e preserva a calibração conjunta
                over_mask = (row_sums > 1.0 + 1e-3).ravel()
                if np.any(over_mask):
                    block[over_mask] = (
                        block[over_mask] / row_sums[over_mask]
                    ).astype(np.float32)
            else:
                block_drift = float(np.max(np.abs(row_sums - 1.0)))
                if block_drift > max_drift:
                    max_drift = block_drift
                block[:] = (block / row_sums).astype(np.float32)

        if total_non_finite > 0:
            warnings_list.append(
                f"{total_non_finite} amostras com NaN/Inf substituídas por distribuição uniforme."
            )
        if total_zero_sum > 0:
            warnings_list.append(
                f"{total_zero_sum} amostras com soma nula substituídas por distribuição uniforme."
            )
        if not self.allow_sub_stochastic and max_drift > 1e-3:
            warnings_list.append(
                f"Deriva de normalização (max |sum(p)-1| = {max_drift:.5f}) corrigida."
            )

        return y_true_arr, y_prob_arr, warnings_list

    def evaluate_arrays(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.floating[Any]],
        confidence_threshold: float = 0.0,
    ) -> EvaluationReport:
        if not (0.0 <= confidence_threshold < 1.0):
            raise ValueError("confidence_threshold deve estar em [0.0, 1.0).")

        y_true_clean, y_prob_clean, warnings_list = self._validate_and_sanitize_inplace(
            y_true, y_prob
        )
        num_samples = y_true_clean.shape[0]
        k_eff = min(5, self.num_classes)

        y_pred = np.empty(num_samples, dtype=np.int64)
        top5_hits = 0
        nll_sum = 0.0

        for start in range(0, num_samples, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, num_samples)
            block = y_prob_clean[start:end]
            yt_block = y_true_clean[start:end]

            max_probs = np.max(block, axis=1)
            raw_preds = np.argmax(block, axis=1)
            y_pred[start:end] = np.where(max_probs >= confidence_threshold, raw_preds, -1)

            top_k_idx = np.argpartition(block, -k_eff, axis=1)[:, -k_eff:]
            top5_hits += int(np.sum(np.any(top_k_idx == yt_block[:, None], axis=1)))

            true_class_probs = np.clip(
                block[np.arange(end - start), yt_block].astype(np.float64),
                1e-15,
                1.0 - 1e-15,
            )
            nll_sum -= float(np.sum(np.log(true_class_probs)))

        rejected_count = int(np.sum(y_pred == -1))
        top1_acc = float(np.mean(y_pred == y_true_clean))
        top5_acc = float(top5_hits / num_samples)
        ll_value = float(nll_sum / num_samples)

        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true_clean, y_pred, labels=self.all_labels, average=None, zero_division=0
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_clean, y_pred, labels=self.all_labels, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_clean, y_pred, labels=self.all_labels, average="weighted", zero_division=0
        )
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_clean, y_pred, labels=self.all_labels, average="micro", zero_division=0
        )

        ap_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)
        auc_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)

        for c in self.all_labels:
            y_true_bin = (y_true_clean == c).astype(np.uint8)
            pos_count = int(np.sum(y_true_bin))
            class_name = self.id_to_name[c]

            if pos_count == 0:
                warnings_list.append(
                    f"Classe '{class_name}' (id={c}) possui support=0; excluída de mAP/ROC-AUC macro."
                )
                continue

            col_scores = np.ascontiguousarray(y_prob_clean[:, c], dtype=np.float32)
            ap_per_class[c] = float(average_precision_score(y_true_bin, col_scores))

            if pos_count == num_samples:
                warnings_list.append(
                    f"Classe '{class_name}' (id={c}) possui 100% das amostras; ROC-AUC indefinido."
                )
            else:
                auc_per_class[c] = float(roc_auc_score(y_true_bin, col_scores))

        supports_arr = np.asarray(support_c, dtype=np.float64)

        valid_ap = ~np.isnan(ap_per_class)
        map_macro = float(np.mean(ap_per_class[valid_ap])) if np.any(valid_ap) else 0.0
        map_weighted = (
            float(np.average(ap_per_class[valid_ap], weights=supports_arr[valid_ap]))
            if np.any(valid_ap) and np.sum(supports_arr[valid_ap]) > 0
            else 0.0
        )

        valid_auc = ~np.isnan(auc_per_class)
        roc_auc_macro = (
            float(np.mean(auc_per_class[valid_auc])) if np.any(valid_auc) else 0.0
        )
        roc_auc_weighted = (
            float(np.average(auc_per_class[valid_auc], weights=supports_arr[valid_auc]))
            if np.any(valid_auc) and np.sum(supports_arr[valid_auc]) > 0
            else 0.0
        )

        cm = confusion_matrix(y_true_clean, y_pred, labels=self.all_labels).astype(np.int64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float64),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        )

        pred_counts = np.bincount(y_pred[y_pred >= 0], minlength=self.num_classes)

        per_class_report: dict[str, ClassMetrics] = {}
        for c in self.all_labels:
            cname = self.id_to_name[c]
            per_class_report[cname] = ClassMetrics(
                class_id=c,
                class_name=cname,
                support=int(support_c[c]),
                predicted_count=int(pred_counts[c]),
                precision=float(prec_c[c]),
                recall=float(rec_c[c]),
                f1_score=float(f1_c[c]),
                average_precision=float(ap_per_class[c]),
                roc_auc=float(auc_per_class[c]),
            )

        _release_memory()

        return EvaluationReport(
            level_name=self.level_name,
            num_samples=num_samples,
            num_classes=self.num_classes,
            confidence_threshold=confidence_threshold,
            rejected_count=rejected_count,
            top1_accuracy=top1_acc,
            top5_accuracy=top5_acc,
            log_loss_value=ll_value,
            map_macro=map_macro,
            map_weighted=map_weighted,
            precision_macro=float(prec_macro),
            precision_weighted=float(prec_weighted),
            precision_micro=float(prec_micro),
            recall_macro=float(rec_macro),
            recall_weighted=float(rec_weighted),
            recall_micro=float(rec_micro),
            f1_macro=float(f1_macro),
            f1_weighted=float(f1_weighted),
            f1_micro=float(f1_micro),
            roc_auc_macro=roc_auc_macro,
            roc_auc_weighted=roc_auc_weighted,
            confusion_matrix=cm,
            confusion_matrix_normalized=cm_norm,
            per_class=per_class_report,
            warnings=warnings_list,
        )


# ==============================================================================
# Extrator Hierárquico (Ordem -> Família) com Controle de Memória
# ==============================================================================


class HierarchicalPredictionExtractor:
    """
    Orquestra a inferência hierárquica em duas camadas:
      1. Executa `classificador_ordem` em chunks.
      2. Roteia as amostras para os modelos de `classificador_familia` (um por vez),
         calculando a distribuição conjunta P(Ordem, Família | x).
    """

    def __init__(self, hierarchical_model: Any) -> None:
        self._validate_hierarchical_container(hierarchical_model)
        self.h_model = hierarchical_model
        self.order_model = hierarchical_model.classificador_ordem
        self.family_models: Mapping[str, Any] = hierarchical_model.classificador_familia

        self.order_id_to_name: dict[int, str] = {
            int(k): str(v) for k, v in self.order_model.names.items()
        }
        if set(self.order_id_to_name.keys()) != set(range(len(self.order_id_to_name))):
            raise LabelAlignmentError(
                "Os índices em classificador_ordem.names devem ser contíguos [0..C-1]."
            )
        self.order_name_to_id: dict[str, int] = {
            v: k for k, v in self.order_id_to_name.items()
        }

        missing_orders = set(self.order_name_to_id.keys()) - set(self.family_models.keys())
        if missing_orders:
            raise LabelAlignmentError(
                f"Ordens presentes em classificador_ordem.names mas ausentes no dicionário "
                f"classificador_familia: {sorted(missing_orders)}"
            )

    @staticmethod
    def _validate_yolo_model(model: Any, label: str) -> None:
        if not hasattr(model, "predict") or not hasattr(model, "names") or not model.names:
            raise EvaluationError(f"Modelo inválido para '{label}'.")
        task = getattr(model, "task", None)
        if task is not None and task != "classify":
            raise EvaluationError(
                f"Tarefa incompatível em '{label}': esperado 'classify', recebido '{task}'."
            )

    def _validate_hierarchical_container(self, h_model: Any) -> None:
        if not hasattr(h_model, "classificador_ordem") or not hasattr(
            h_model, "classificador_familia"
        ):
            raise EvaluationError(
                "Objeto inválido: esperado atributos 'classificador_ordem' e 'classificador_familia'."
            )
        if h_model.classificador_ordem is None:
            raise EvaluationError("'classificador_ordem' não pode ser None.")
        if not isinstance(h_model.classificador_familia, Mapping):
            raise EvaluationError("'classificador_familia' deve ser um dicionário.")

        self._validate_yolo_model(h_model.classificador_ordem, "classificador_ordem")
        for order_name, fam_model in h_model.classificador_familia.items():
            if fam_model is not None:
                self._validate_yolo_model(fam_model, f"classificador_familia['{order_name}']")

    @staticmethod
    def _scan_images_fast(directory: str) -> list[str]:
        found: list[str] = []
        stack: list[str] = [directory]
        while stack:
            current_dir = stack.pop()
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in VALID_IMAGE_EXTENSIONS:
                            found.append(entry.path)
        found.sort()
        return found

    def resolve_hierarchical_manifest(
        self,
        val_dir: str | Path,
    ) -> tuple[
        list[str],
        NDArray[np.int64],
        NDArray[np.int64],
        dict[int, str],
        dict[int, list[int]],
    ]:
        """
        Lê a estrutura `val_dir/<Ordem>/val/<Familia>/*` (com fallback para
        `val_dir/<Ordem>/<Familia>/*`) e constrói a taxonomia global unificada.
        """
        root_str = os.fspath(val_dir)
        if not os.path.isdir(root_str):
            raise FileNotFoundError(f"Diretório de validação não encontrado: {root_str}")

        # 1. Descobre pastas presentes em disco agrupadas por Ordem
        discovered_files: dict[str, dict[str, list[str]]] = {}
        with os.scandir(root_str) as order_entries:
            order_dirs = sorted(
                (e for e in order_entries if e.is_dir(follow_symlinks=False)),
                key=lambda e: e.name,
            )

        for order_entry in order_dirs:
            order_name = order_entry.name
            if order_name not in self.order_name_to_id:
                raise LabelAlignmentError(
                    f"Diretório de ordem '{order_name}' não existe em classificador_ordem.names."
                )

            # Suporta `val_dir/<Ordem>/val/<Familia>` e fallback `val_dir/<Ordem>/<Familia>`
            nested_val = os.path.join(order_entry.path, "val")
            fam_root = nested_val if os.path.isdir(nested_val) else order_entry.path

            with os.scandir(fam_root) as fam_entries:
                fam_dirs = sorted(
                    (e for e in fam_entries if e.is_dir(follow_symlinks=False)),
                    key=lambda e: e.name,
                )

            discovered_files[order_name] = {}
            for fam_entry in fam_dirs:
                imgs = self._scan_images_fast(fam_entry.path)
                discovered_files[order_name][fam_entry.name] = imgs

        # 2. Constrói o registro global de Famílias alinhado aos modelos carregados
        global_fam_id_to_name: dict[int, str] = {}
        global_pair_to_id: dict[tuple[str, str], int] = {}
        order_to_global_fam_ids: dict[int, list[int]] = {}
        next_global_id = 0

        for order_id in range(len(self.order_id_to_name)):
            order_name = self.order_id_to_name[order_id]
            fam_model = self.family_models.get(order_name)
            disk_fam_map = discovered_files.get(order_name, {})
            disk_fam_names = list(disk_fam_map.keys())

            if fam_model is not None:
                sorted_keys = sorted(int(k) for k in fam_model.names.keys())
                if sorted_keys != list(range(len(sorted_keys))):
                    raise LabelAlignmentError(
                        f"Índices não contíguos em classificador_familia['{order_name}'].names."
                    )
                model_fam_names = [str(fam_model.names[k]) for k in sorted_keys]

                unknown_on_disk = set(disk_fam_names) - set(model_fam_names)
                if unknown_on_disk:
                    raise LabelAlignmentError(
                        f"Famílias {sorted(unknown_on_disk)} em '{order_name}' não existem "
                        f"em classificador_familia['{order_name}'].names: {model_fam_names}"
                    )

                g_ids: list[int] = []
                for fname in model_fam_names:
                    global_fam_id_to_name[next_global_id] = f"{order_name}/{fname}"
                    global_pair_to_id[(order_name, fname)] = next_global_id
                    g_ids.append(next_global_id)
                    next_global_id += 1
                order_to_global_fam_ids[order_id] = g_ids

            else:
                # Modelo da Camada 2 é None
                if len(disk_fam_names) == 1:
                    fname = disk_fam_names[0]
                    global_fam_id_to_name[next_global_id] = f"{order_name}/{fname}"
                    global_pair_to_id[(order_name, fname)] = next_global_id
                    order_to_global_fam_ids[order_id] = [next_global_id]
                    next_global_id += 1
                elif len(disk_fam_names) > 1:
                    raise EvaluationError(
                        f"A ordem '{order_name}' possui {len(disk_fam_names)} famílias no disco "
                        f"({disk_fam_names}), mas classificador_familia['{order_name}'] é None."
                    )
                else:
                    # Ordem sem modelo Camada 2 e sem pastas no val_dir: cria nó folha placeholder
                    placeholder = f"{order_name}/(única)"
                    global_fam_id_to_name[next_global_id] = placeholder
                    order_to_global_fam_ids[order_id] = [next_global_id]
                    next_global_id += 1

        # 3. Monta arrays finais de caminhos e ground-truths (Ordem e Família)
        image_paths: list[str] = []
        y_true_order_list: list[int] = []
        y_true_family_list: list[int] = []

        for order_name, fam_dict in discovered_files.items():
            order_id = self.order_name_to_id[order_name]
            for fam_name, imgs in fam_dict.items():
                global_fam_id = global_pair_to_id[(order_name, fam_name)]
                image_paths.extend(imgs)
                y_true_order_list.extend([order_id] * len(imgs))
                y_true_family_list.extend([global_fam_id] * len(imgs))

        if not image_paths:
            raise LabelAlignmentError(f"Nenhuma imagem válida encontrada em '{root_str}'.")

        y_true_order = np.array(y_true_order_list, dtype=np.int64)
        y_true_family = np.array(y_true_family_list, dtype=np.int64)
        del y_true_order_list, y_true_family_list

        return (
            image_paths,
            y_true_order,
            y_true_family,
            global_fam_id_to_name,
            order_to_global_fam_ids,
        )

    @staticmethod
    def _extract_probs_vector(probs_obj: Any, expected_dim: int) -> NDArray[np.float32]:
        if probs_obj is None or not hasattr(probs_obj, "data"):
            raise EvaluationError("Resultado da predição não contém 'probs.data'.")
        raw = probs_obj.data
        if torch is not None and isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy()
        elif hasattr(raw, "detach"):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)

        arr_f32 = arr.astype(np.float32, copy=False)
        if arr_f32.shape != (expected_dim,):
            raise EvaluationError(
                f"Shape inesperado em probs.data: esperado ({expected_dim},), obtido {arr_f32.shape}."
            )
        return arr_f32

    def _run_model_chunked(
        self,
        model: Any,
        paths: Sequence[str],
        num_classes: int,
        batch_size: int,
        chunk_size: int,
        imgsz: int | None,
        device: str | int | None,
        quantize: int | str | None,
        on_prediction: Any,
    ) -> None:
        """Executa model.predict em chunks e invoca callback `on_prediction(global_idx, probs_vec)`."""
        base_kwargs: dict[str, Any] = {
            "stream": True,
            "batch": batch_size,
            "verbose": False,
            "save": False,
            "plots": False,
        }
        if imgsz is not None:
            base_kwargs["imgsz"] = imgsz
        if device is not None:
            base_kwargs["device"] = device
        if quantize is not None:
            base_kwargs["quantize"] = quantize

        total = len(paths)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk_paths = paths[start:end]
            expected_len = end - start

            results_gen = None
            processed = 0
            try:
                if torch is not None:
                    with torch.inference_mode():
                        results_gen = model.predict(source=chunk_paths, **base_kwargs)
                        for offset, res in enumerate(results_gen):
                            if offset >= expected_len:
                                raise EvaluationError("Excesso de predições retornadas no chunk.")
                            vec = self._extract_probs_vector(res.probs, num_classes)
                            on_prediction(start + offset, vec)
                            del res
                            processed += 1
                else:
                    results_gen = model.predict(source=chunk_paths, **base_kwargs)
                    for offset, res in enumerate(results_gen):
                        if offset >= expected_len:
                            raise EvaluationError("Excesso de predições retornadas no chunk.")
                        vec = self._extract_probs_vector(res.probs, num_classes)
                        on_prediction(start + offset, vec)
                        del res
                        processed += 1
            finally:
                del results_gen
                del chunk_paths
                _release_memory()

            if processed != expected_len:
                raise EvaluationError(
                    f"Desalinhamento no chunk [{start}:{end}]: esperadas {expected_len}, "
                    f"processadas {processed}."
                )

    def extract(
        self,
        val_dir: str | Path,
        batch_size: int = 16,
        chunk_size: int = 256,
        top_k_routing: int = 1,
        imgsz: int | None = None,
        device: str | int | None = None,
        quantize: int | str | None = None,
        use_memmap: bool = False,
        memmap_dir: str | Path | None = None,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32],
        dict[int, str],
    ]:
        """
        Executa a inferência completa da hierarquia retornando:
        `(y_true_order, y_prob_order, y_true_family, y_prob_family, global_fam_id_to_name)`.
        """
        num_orders = len(self.order_id_to_name)
        if top_k_routing < 1:
            raise ValueError("top_k_routing deve ser >= 1.")
        k_route = min(top_k_routing, num_orders)

        (
            image_paths,
            y_true_order,
            y_true_family,
            global_fam_id_to_name,
            order_to_global_fam_ids,
        ) = self.resolve_hierarchical_manifest(val_dir)

        num_samples = len(image_paths)
        num_families = len(global_fam_id_to_name)

        # ----------------------------------------------------------------------
        # CAMADA 1: Predição das Ordens
        # ----------------------------------------------------------------------
        y_prob_order = np.empty((num_samples, num_orders), dtype=np.float32)

        def _store_order_pred(idx: int, vec: NDArray[np.float32]) -> None:
            y_prob_order[idx] = vec

        self._run_model_chunked(
            model=self.order_model,
            paths=image_paths,
            num_classes=num_orders,
            batch_size=batch_size,
            chunk_size=chunk_size,
            imgsz=imgsz,
            device=device,
            quantize=quantize,
            on_prediction=_store_order_pred,
        )

        # Define quais ordens serão ativadas para cada imagem (Top-K Routing)
        if k_route == 1:
            routed_orders = np.argmax(y_prob_order, axis=1)[:, None]  # (N, 1)
        elif k_route < num_orders:
            routed_orders = np.argpartition(y_prob_order, -k_route, axis=1)[:, -k_route:]
        else:
            routed_orders = np.broadcast_to(
                np.arange(num_orders, dtype=np.int64), (num_samples, num_orders)
            )

        # ----------------------------------------------------------------------
        # CAMADA 2: Predição das Famílias (uma Ordem por vez para poupar RAM/VRAM)
        # ----------------------------------------------------------------------
        if use_memmap:
            base_dir = os.fspath(memmap_dir) if memmap_dir else tempfile.gettempdir()
            tmp_file = tempfile.NamedTemporaryFile(
                prefix="yolo_hier_fam_", suffix=".dat", dir=base_dir, delete=False
            )
            tmp_path = tmp_file.name
            tmp_file.close()
            y_prob_family: NDArray[np.float32] = np.memmap(
                tmp_path, dtype=np.float32, mode="w+", shape=(num_samples, num_families)
            )
            y_prob_family[:] = 0.0
        else:
            y_prob_family = np.zeros((num_samples, num_families), dtype=np.float32)

        for order_id in range(num_orders):
            sample_indices = np.flatnonzero(np.any(routed_orders == order_id, axis=1))
            if sample_indices.size == 0:
                continue

            order_name = self.order_id_to_name[order_id]
            fam_model = self.family_models.get(order_name)
            global_fam_ids = np.asarray(order_to_global_fam_ids[order_id], dtype=np.intp)
            order_probs_subset = y_prob_order[sample_indices, order_id]

            if fam_model is None or global_fam_ids.size == 1:
                # Ordem monotípica: P(Família | Ordem) = 1.0 -> P(Ordem, Família) = P(Ordem)
                y_prob_family[sample_indices, global_fam_ids[0]] = order_probs_subset
                continue

            routed_paths = [image_paths[int(i)] for i in sample_indices]
            num_local_fams = global_fam_ids.size

            def _store_family_pred(local_idx: int, local_vec: NDArray[np.float32]) -> None:
                g_sample_idx = int(sample_indices[local_idx])
                p_order = np.float32(order_probs_subset[local_idx])
                y_prob_family[g_sample_idx, global_fam_ids] = local_vec * p_order

            self._run_model_chunked(
                model=fam_model,
                paths=routed_paths,
                num_classes=num_local_fams,
                batch_size=batch_size,
                chunk_size=chunk_size,
                imgsz=imgsz,
                device=device,
                quantize=quantize,
                on_prediction=_store_family_pred,
            )

            if isinstance(y_prob_family, np.memmap):
                y_prob_family.flush()

            del routed_paths
            _release_memory()

        del image_paths
        _release_memory()

        return (
            y_true_order,
            y_prob_order,
            y_true_family,
            y_prob_family,
            global_fam_id_to_name,
        )


# ==============================================================================
# Fachada Principal de Alto Nível
# ==============================================================================


def evaluate_hierarchical_classifier(
    hierarchical_model: Any,
    val_dir: str | Path,
    batch_size: int = 16,
    chunk_size: int = 256,
    top_k_routing: int = 1,
    imgsz: int | None = None,
    device: str | int | None = None,
    quantize: int | str | None = None,
    confidence_threshold: float = 0.0,
    strict_nan: bool = True,
    use_memmap: bool = False,
    memmap_dir: str | Path | None = None,
) -> HierarchicalEvaluationReport:
    """
    Avalia o objeto `Classificador_Hierarquico` na estrutura `val_dir/<Ordem>/val/<Familia>/`.

    Args:
        hierarchical_model: Instância com `.classificador_ordem` e `.classificador_familia`.
        val_dir: Diretório raiz de validação.
        batch_size: Tamanho do lote por forward pass (padrão 16).
        chunk_size: Limite de imagens antes de limpar RAM/VRAM (padrão 256).
        top_k_routing: Quantas ordens mais prováveis acionar na Camada 2 (1 = Hard Routing).
        imgsz: Resolução opcional de entrada.
        device: Dispositivo ('cuda:0', 'cpu', etc.).
        quantize: Precisão de inferência (ex.: 16 para FP16, None para FP32).
        confidence_threshold: Limiar mínimo de confiança conjunta para aceitar predição.
        strict_nan: Se True, interrompe em probabilidades degeneradas.
        use_memmap: Se True, grava a matriz global de famílias em disco em vez de RAM.
        memmap_dir: Diretório opcional para o arquivo temporário do memmap.
    """
    extractor = HierarchicalPredictionExtractor(hierarchical_model)
    (
        y_true_order,
        y_prob_order,
        y_true_family,
        y_prob_family,
        global_fam_id_to_name,
    ) = extractor.extract(
        val_dir=val_dir,
        batch_size=batch_size,
        chunk_size=chunk_size,
        top_k_routing=top_k_routing,
        imgsz=imgsz,
        device=device,
        quantize=quantize,
        use_memmap=use_memmap,
        memmap_dir=memmap_dir,
    )

    # 1. Avaliação da Camada 1 (Ordem)
    order_evaluator = ClassificationMetricsEvaluator(
        class_names=extractor.order_id_to_name,
        level_name="Camada 1 - Ordem",
        strict_nan=strict_nan,
        allow_sub_stochastic=False,
    )
    order_report = order_evaluator.evaluate_arrays(
        y_true=y_true_order,
        y_prob=y_prob_order,
        confidence_threshold=confidence_threshold,
    )

    # Acurácia condicional: dentre as imagens em que a Ordem foi acertada, quantas acertaram a Família?
    order_preds = np.argmax(y_prob_order, axis=1)
    correct_order_mask = order_preds == y_true_order
    correct_order_count = int(np.sum(correct_order_mask))

    del y_prob_order
    _release_memory()

    # 2. Avaliação Ponta-a-Ponta (Família Global)
    num_orders = len(extractor.order_id_to_name)
    family_evaluator = ClassificationMetricsEvaluator(
        class_names=global_fam_id_to_name,
        level_name="Camada 2 - Família (Global)",
        strict_nan=strict_nan,
        allow_sub_stochastic=(top_k_routing < num_orders),
    )
    family_report = family_evaluator.evaluate_arrays(
        y_true=y_true_family,
        y_prob=y_prob_family,
        confidence_threshold=confidence_threshold,
    )

    if correct_order_count > 0:
        fam_preds_all = np.argmax(y_prob_family, axis=1)
        cond_acc = float(
            np.mean(fam_preds_all[correct_order_mask] == y_true_family[correct_order_mask])
        )
    else:
        cond_acc = 0.0

    # Remove arquivo temporário do memmap se utilizado
    if isinstance(y_prob_family, np.memmap) and y_prob_family.filename:
        memmap_path = y_prob_family.filename
        del y_prob_family
        _release_memory()
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    return HierarchicalEvaluationReport(
        top_k_routing=top_k_routing,
        order_report=order_report,
        family_report=family_report,
        family_accuracy_given_correct_order=cond_acc,
        correct_order_count=correct_order_count,
    )


# ==============================================================================
# Teste Integrado de Sanidade (Simula Classificador_Hierarquico + Diretórios)
# ==============================================================================


if __name__ == "__main__":
    class Classificador_Hierarquico:
        def __init__(self) -> None:
            self.classificador_ordem: Any = None
            self.classificador_familia: dict[str, Any] = {
                "Thysanoptera": None, "Coleoptera": None,
                "Diptera": None, "Ixodida": None, "Megaloptera": None, "Phasmida": None, "Hymenoptera": None,
                "Psocodea": None, "Araneae": None, "Mesostigmata": None, "Symphyla_order_incertae_sedis": None,
                "Stylommatophora": None, "Julida": None, "Sarcoptiformes": None, "Siphonaptera": None, "Isopoda": None,
                "Polydesmida": None, "Trombidiformes": None, "Orthoptera": None, "Dermaptera": None, "Neuroptera": None, "Odonata": None,
                "Lepidoptera": None, "Hemiptera": None, "Blattodea": None, "Embioptera": None, "Mantodea": None, "Symphypleona": None,
            }

    class _MockProbs:
        def __init__(self, vec: NDArray[np.float32]) -> None:
            self.data = torch.tensor(vec, dtype=torch.float16) if torch is not None else vec

    class _MockResult:
        def __init__(self, vec: NDArray[np.float32]) -> None:
            self.probs = _MockProbs(vec)

    class _MockYOLO:
        def __init__(self, names: dict[int, str], target_bias: dict[str, int]) -> None:
            self.task = "classify"
            self.names = names
            self._bias = target_bias
            self._rng = np.random.default_rng(123)

        def predict(self, source: Sequence[str], stream: bool = True, **kwargs: Any):
            assert stream is True
            c = len(self.names)
            for path_str in source:
                logits = self._rng.normal(0.0, 0.8, size=c)
                for keyword, cls_idx in self._bias.items():
                    if keyword in path_str:
                        logits[cls_idx] += 3.0
                exp_l = np.exp(logits - np.max(logits))
                probs = (exp_l / np.sum(exp_l)).astype(np.float32)
                yield _MockResult(probs)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        # Cria estrutura exata: val_dir/<Ordem>/val/<Familia>/img.jpg
        structure = {
            "Coleoptera": ["Carabidae", "Scarabaeidae"],
            "Diptera": ["Culicidae"],  # Ordem monotípica (modelo família = None)
            "Hymenoptera": ["Formicidae", "Apidae"],
        }
        for order_name, fams in structure.items():
            for fam in fams:
                fam_dir = base / order_name / "val" / fam
                fam_dir.mkdir(parents=True)
                for i in range(40):
                    (fam_dir / f"{order_name}_{fam}_{i}.jpg").touch()

        hier = Classificador_Hierarquico()
        all_orders = list(hier.classificador_familia.keys())
        order_names_map = {i: name for i, name in enumerate(all_orders)}

        hier.classificador_ordem = _MockYOLO(
            names=order_names_map,
            target_bias={name: i for i, name in order_names_map.items()},
        )
        hier.classificador_familia["Coleoptera"] = _MockYOLO(
            names={0: "Carabidae", 1: "Scarabaeidae"},
            target_bias={"Carabidae": 0, "Scarabaeidae": 1},
        )
        hier.classificador_familia["Hymenoptera"] = _MockYOLO(
            names={0: "Apidae", 1: "Formicidae"},
            target_bias={"Apidae": 0, "Formicidae": 1},
        )
        # Note que hier.classificador_familia["Diptera"] permanece None propositalmente!

        report = evaluate_hierarchical_classifier(
            hierarchical_model=hier,
            val_dir=base,
            batch_size=16,
            chunk_size=32,
            top_k_routing=1,
            quantize=16,
            use_memmap=True,
        )
        print(report.format_summary())