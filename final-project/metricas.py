"""
Módulo otimizado para baixa pegada de memória (RAM/VRAM) na avaliação de
modelos Ultralytics YOLO-cls em datasets de larga escala.
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
    """Exceção base para falhas no pipeline de avaliação de classificação."""


class DegeneratePredictionError(EvaluationError):
    """Lançada quando o modelo produz probabilidades NaN, Inf ou nulas."""


class LabelAlignmentError(EvaluationError):
    """Lançada quando há inconsistência entre ground-truth e model.names."""


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
    """Relatório consolidado e serializável da avaliação do modelo."""

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

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def format_summary(self) -> str:
        lines = [
            "=" * 88,
            f"RELATÓRIO DE AVALIAÇÃO YOLO-CLS | Amostras: {self.num_samples} | Classes: {self.num_classes}",
            f"Threshold de Confiança: {self.confidence_threshold:.2f} | Rejeitadas: {self.rejected_count}",
            "=" * 88,
            f"Top-1 Accuracy : {self.top1_accuracy:.4f} | Top-5 Accuracy : {self.top5_accuracy:.4f}",
            f"mAP (Macro)    : {self.map_macro:.4f} | mAP (Weighted) : {self.map_weighted:.4f}",
            f"F1 (Macro)     : {self.f1_macro:.4f} | F1 (Weighted)  : {self.f1_weighted:.4f} | F1 (Micro): {self.f1_micro:.4f}",
            f"Precision Macro: {self.precision_macro:.4f} | Recall Macro   : {self.recall_macro:.4f}",
            f"ROC-AUC (Macro): {self.roc_auc_macro:.4f} | Log-Loss       : {self.log_loss_value:.4f}",
            "-" * 88,
            f"{'ID':<4} {'Classe':<24} {'Supp':>6} {'Pred':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7} {'AUC':>7}",
            "-" * 88,
        ]
        for cm in self.per_class.values():
            ap_str = f"{cm.average_precision:.4f}" if not np.isnan(cm.average_precision) else "   N/A"
            auc_str = f"{cm.roc_auc:.4f}" if not np.isnan(cm.roc_auc) else "   N/A"
            lines.append(
                f"{cm.class_id:<4} {cm.class_name[:24]:<24} {cm.support:>6} {cm.predicted_count:>6} "
                f"{cm.precision:>7.4f} {cm.recall:>7.4f} {cm.f1_score:>7.4f} {ap_str:>7} {auc_str:>7}"
            )
        if self.warnings:
            lines.append("-" * 88)
            lines.append("ALERTAS DE VALIDAÇÃO:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        lines.append("=" * 88)
        return "\n".join(lines)


# ==============================================================================
# Utilitário de Limpeza de Memória
# ==============================================================================


def _release_memory() -> None:
    """Força coleta de lixo no Python e libera cache ocioso da GPU."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# Extrator Desacoplado com Chunking e Streaming
# ==============================================================================


class YOLOPredictionExtractor:
    """
    Extrai probabilidades do Ultralytics YOLO-cls em chunks limitados com
    limpeza explícita de memória entre blocos para evitar OOM e Swap.
    """

    def __init__(self, model: Any) -> None:
        self._validate_model(model)
        self.model = model
        self.id_to_name: dict[int, str] = {
            int(k): str(v) for k, v in model.names.items()
        }
        self._verify_contiguous_indices()
        self.name_to_id: dict[str, int] = {
            v: k for k, v in self.id_to_name.items()
        }

    @staticmethod
    def _validate_model(model: Any) -> None:
        if not hasattr(model, "predict") or not hasattr(model, "names"):
            raise EvaluationError(
                "Objeto 'model' inválido: esperado uma instância Ultralytics YOLO."
            )
        task = getattr(model, "task", None)
        if task is not None and task != "classify":
            raise EvaluationError(
                f"Tarefa incompatível: esperado 'classify', recebido '{task}'."
            )
        if not model.names:
            raise EvaluationError("O modelo carregado não possui 'model.names' populado.")

    def _verify_contiguous_indices(self) -> None:
        expected = set(range(len(self.id_to_name)))
        actual = set(self.id_to_name.keys())
        if actual != expected:
            raise LabelAlignmentError(
                f"Índices de classes em model.names não são contíguos [0..C-1]: {sorted(actual)}"
            )

    @staticmethod
    def _scan_images_fast(directory: str) -> list[str]:
        """Varredura recursiva leve com os.scandir (evita overhead de objetos Path)."""
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

    def resolve_dataset_manifest(
        self,
        data_source: str | Path | Sequence[tuple[str | Path, int | str]],
    ) -> tuple[list[str], NDArray[np.int64]]:
        """
        Retorna listas paralelas `(image_paths, y_true)` minimizando objetos em RAM.
        """
        image_paths: list[str] = []
        labels: list[int] = []

        if isinstance(data_source, (str, Path)):
            root_str = os.fspath(data_source)
            if not os.path.isdir(root_str):
                raise FileNotFoundError(f"Diretório de validação não encontrado: {root_str}")

            with os.scandir(root_str) as entries:
                class_dirs = sorted(
                    (e for e in entries if e.is_dir(follow_symlinks=False)),
                    key=lambda e: e.name,
                )

            for class_entry in class_dirs:
                class_name = class_entry.name
                if class_name not in self.name_to_id:
                    raise LabelAlignmentError(
                        f"Diretório '{class_name}' não existe em model.names: "
                        f"{list(self.name_to_id.keys())}"
                    )
                class_id = self.name_to_id[class_name]
                class_files = self._scan_images_fast(class_entry.path)
                image_paths.extend(class_files)
                labels.extend([class_id] * len(class_files))
        else:
            for img_path, label in data_source:
                if isinstance(label, str):
                    if label not in self.name_to_id:
                        raise LabelAlignmentError(
                            f"Rótulo '{label}' ausente em model.names."
                        )
                    class_id = self.name_to_id[label]
                elif isinstance(label, (int, np.integer)):
                    class_id = int(label)
                    if class_id not in self.id_to_name:
                        raise LabelAlignmentError(
                            f"ID de classe {class_id} fora do intervalo de model.names."
                        )
                else:
                    raise LabelAlignmentError(f"Tipo de rótulo não suportado: {type(label)}")
                image_paths.append(os.fspath(img_path))
                labels.append(class_id)

        if not image_paths:
            raise LabelAlignmentError("Nenhuma imagem válida encontrada na fonte de dados.")

        y_true = np.array(labels, dtype=np.int64)
        del labels
        return image_paths, y_true

    @staticmethod
    def _write_probs_to_slot(probs_obj: Any, target_slot: NDArray[np.float32]) -> None:
        """Extrai probs.data e grava diretamente no slice pré-alocado em float32."""
        if probs_obj is None or not hasattr(probs_obj, "data"):
            raise EvaluationError(
                "Resultado da predição não contém atributo 'probs.data'."
            )
        raw = probs_obj.data
        if torch is not None and isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy()
        elif hasattr(raw, "detach"):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)

        if arr.shape != target_slot.shape:
            raise EvaluationError(
                f"Shape inesperado em probs.data: esperado {target_slot.shape}, obtido {arr.shape}."
            )
        target_slot[:] = arr

    def extract(
        self,
        data_source: str | Path | Sequence[tuple[str | Path, int | str]],
        batch_size: int = 16,
        chunk_size: int = 512,
        imgsz: int | None = None,
        device: str | int | None = None,
        quantize: int | str | None = None,
        use_memmap: bool = False,
        memmap_dir: str | Path | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """
        Executa inferência em chunks controlados para impedir crescimento de RAM/Swap.

        Args:
            data_source: Diretório raiz por classe ou lista de `(caminho, classe)`.
            batch_size: Imagens por forward pass (padrão 16 para baixo consumo).
            chunk_size: Imagens processadas antes de resetar o gerador do YOLO e liberar RAM.
            imgsz: Resolução opcional de entrada.
            device: Dispositivo ('cuda:0', 'cpu', etc.).
            half: Ativa FP16 na inferência.
            use_memmap: Se True, armazena y_prob em arquivo mapeado em disco em vez da RAM.
            memmap_dir: Diretório opcional para o arquivo memmap.
        """
        if batch_size < 1 or chunk_size < 1:
            raise ValueError("batch_size e chunk_size devem ser >= 1.")

        image_paths, y_true = self.resolve_dataset_manifest(data_source)
        num_samples = len(image_paths)
        num_classes = len(self.id_to_name)

        if use_memmap:
            base_dir = os.fspath(memmap_dir) if memmap_dir else tempfile.gettempdir()
            tmp_file = tempfile.NamedTemporaryFile(
                prefix="yolo_probs_", suffix=".dat", dir=base_dir, delete=False
            )
            tmp_path = tmp_file.name
            tmp_file.close()
            y_prob: NDArray[np.float32] = np.memmap(
                tmp_path, dtype=np.float32, mode="w+", shape=(num_samples, num_classes)
            )
        else:
            y_prob = np.empty((num_samples, num_classes), dtype=np.float32)

        base_predict_kwargs: dict[str, Any] = {
            "stream": True,
            "batch": batch_size,
            "verbose": False,
            "save": False,
            "plots": False,
            
        }
        if imgsz is not None:
            base_predict_kwargs["imgsz"] = imgsz
        if device is not None:
            base_predict_kwargs["device"] = device
        if quantize is not None:
            base_predict_kwargs["quantize"] = quantize

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk_paths = image_paths[start:end]
            expected_len = end - start

            results_generator = None
            processed = 0
            try:
                if torch is not None:
                    with torch.inference_mode():
                        results_generator = self.model.predict(
                            source=chunk_paths, **base_predict_kwargs
                        )
                        for offset, result in enumerate(results_generator):
                            if offset >= expected_len:
                                raise EvaluationError(
                                    "O modelo retornou mais predições do que imagens no chunk."
                                )
                            self._write_probs_to_slot(result.probs, y_prob[start + offset])
                            del result
                            processed += 1
                else:
                    results_generator = self.model.predict(
                        source=chunk_paths, **base_predict_kwargs
                    )
                    for offset, result in enumerate(results_generator):
                        if offset >= expected_len:
                            raise EvaluationError(
                                "O modelo retornou mais predições do que imagens no chunk."
                            )
                        self._write_probs_to_slot(result.probs, y_prob[start + offset])
                        del result
                        processed += 1
            finally:
                del results_generator
                del chunk_paths
                _release_memory()

            if processed != expected_len:
                raise EvaluationError(
                    f"Desalinhamento no chunk [{start}:{end}]: enviadas {expected_len}, "
                    f"processadas {processed}."
                )

            if isinstance(y_prob, np.memmap):
                y_prob.flush()

        del image_paths
        _release_memory()
        return y_true, y_prob


# ==============================================================================
# Motor Matemático de Avaliação (Zero-Copy / Processamento em Blocos)
# ==============================================================================


class ClassificationMetricsEvaluator:
    """
    Motor determinístico que calcula todas as métricas processando blocos de
    linhas para evitar duplicação da matriz de probabilidades na RAM.
    """

    def __init__(
        self,
        class_names: Mapping[int, str],
        eps: float = 1e-7,
        strict_nan: bool = True,
        eval_chunk_size: int = 4096,
    ) -> None:
        if not class_names:
            raise ValueError("class_names não pode ser vazio.")
        self.id_to_name: dict[int, str] = {int(k): str(v) for k, v in class_names.items()}
        self.num_classes: int = len(self.id_to_name)
        self.all_labels: list[int] = list(range(self.num_classes))
        if set(self.id_to_name.keys()) != set(self.all_labels):
            raise LabelAlignmentError("As chaves de class_names devem ser 0..C-1.")
        self.eps = eps
        self.strict_nan = strict_nan
        self.eval_chunk_size = eval_chunk_size

    def _validate_and_sanitize_inplace(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], list[str]]:
        warnings_list: list[str] = []

        y_true_arr = np.asarray(y_true, dtype=np.int64)
        # Evita cópia se já estiver em float32 gravável
        if isinstance(y_prob, np.ndarray) and y_prob.dtype == np.float32 and y_prob.flags.writeable:
            y_prob_arr = y_prob
        else:
            y_prob_arr = np.array(y_prob, dtype=np.float32, copy=True)

        if y_true_arr.ndim != 1:
            raise ValueError(f"y_true deve ser 1D, recebido shape {y_true_arr.shape}.")
        if y_prob_arr.ndim != 2:
            raise ValueError(f"y_prob deve ser 2D, recebido shape {y_prob_arr.shape}.")
        if y_true_arr.shape[0] == 0:
            raise ValueError("O conjunto de avaliação está vazio (0 amostras).")
        if y_true_arr.shape[0] != y_prob_arr.shape[0]:
            raise ValueError(
                f"Incompatibilidade de amostras: y_true={y_true_arr.shape[0]} vs "
                f"y_prob={y_prob_arr.shape[0]}."
            )
        if y_prob_arr.shape[1] != self.num_classes:
            raise ValueError(
                f"Incompatibilidade de classes: esperado {self.num_classes}, "
                f"recebido {y_prob_arr.shape[1]}."
            )

        if np.any((y_true_arr < 0) | (y_true_arr >= self.num_classes)):
            raise LabelAlignmentError(
                f"y_true contém índices fora do intervalo [0, {self.num_classes - 1}]."
            )

        num_samples = y_true_arr.shape[0]
        total_non_finite = 0
        total_zero_sum = 0
        max_drift = 0.0

        # Sanitização por blocos (sem alocar máscaras globais N x C)
        for start in range(0, num_samples, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, num_samples)
            block = y_prob_arr[start:end]

            non_finite_rows = ~np.all(np.isfinite(block), axis=1)
            if np.any(non_finite_rows):
                bad_count = int(np.sum(non_finite_rows))
                if self.strict_nan:
                    raise DegeneratePredictionError(
                        f"Detectadas amostras com probabilidades NaN ou Inf (bloco {start}:{end})."
                    )
                total_non_finite += bad_count
                block[non_finite_rows] = np.float32(1.0 / self.num_classes)

            if np.any(block < -self.eps):
                raise DegeneratePredictionError(
                    "Detectadas probabilidades negativas (possível saída de logits sem Softmax)."
                )
            np.clip(block, 0.0, None, out=block)

            row_sums = np.sum(block, axis=1, keepdims=True, dtype=np.float64)
            zero_rows = (row_sums <= self.eps).ravel()
            if np.any(zero_rows):
                bad_zero = int(np.sum(zero_rows))
                if self.strict_nan:
                    raise DegeneratePredictionError(
                        "Detectadas amostras com soma de probabilidades igual a zero."
                    )
                total_zero_sum += bad_zero
                block[zero_rows] = np.float32(1.0 / self.num_classes)
                row_sums = np.sum(block, axis=1, keepdims=True, dtype=np.float64)

            block_drift = float(np.max(np.abs(row_sums - 1.0)))
            if block_drift > max_drift:
                max_drift = block_drift

            block[:] = (block / row_sums).astype(np.float32)

        if total_non_finite > 0:
            warnings_list.append(
                f"Detectadas {total_non_finite} amostras com NaN/Inf. Substituídas por distribuição uniforme."
            )
        if total_zero_sum > 0:
            warnings_list.append(
                f"Detectadas {total_zero_sum} amostras com soma nula. Substituídas por distribuição uniforme."
            )
        if max_drift > 1e-3:
            warnings_list.append(
                f"Deriva de normalização detectada (max |sum(p)-1| = {max_drift:.5f}). "
                "Probabilidades foram renormalizadas."
            )

        return y_true_arr, y_prob_arr, warnings_list

    def evaluate_arrays(
        self,
        y_true: NDArray[np.int64],
        y_prob: NDArray[np.floating[Any]],
        confidence_threshold: float = 0.0,
    ) -> EvaluationReport:
        """
        Calcula todas as métricas com alocação mínima de memória auxiliar.
        """
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

        # Passada única em blocos para Top-1, Top-5 e Log-Loss (O(chunk_size * C) RAM)
        for start in range(0, num_samples, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, num_samples)
            block = y_prob_clean[start:end]
            yt_block = y_true_clean[start:end]

            # Top-1 e limiar de rejeição
            max_probs = np.max(block, axis=1)
            raw_preds = np.argmax(block, axis=1)
            y_pred[start:end] = np.where(max_probs >= confidence_threshold, raw_preds, -1)

            # Top-5
            top_k_idx = np.argpartition(block, -k_eff, axis=1)[:, -k_eff:]
            top5_hits += int(np.sum(np.any(top_k_idx == yt_block[:, None], axis=1)))

            # Log-Loss exato no bloco (sem alocar matriz N x C temporária)
            block_f64 = np.clip(block.astype(np.float64), 1e-15, 1.0 - 1e-15)
            block_f64 /= np.sum(block_f64, axis=1, keepdims=True)
            true_class_probs = block_f64[np.arange(end - start), yt_block]
            nll_sum -= float(np.sum(np.log(true_class_probs)))

        rejected_count = int(np.sum(y_pred == -1))
        top1_acc = float(np.mean(y_pred == y_true_clean))
        top5_acc = float(top5_hits / num_samples)
        ll_value = float(nll_sum / num_samples)

        # Precision, Recall, F1
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true_clean,
            y_pred,
            labels=self.all_labels,
            average=None,
            zero_division=0,
        )
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_clean,
            y_pred,
            labels=self.all_labels,
            average="macro",
            zero_division=0,
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_clean,
            y_pred,
            labels=self.all_labels,
            average="weighted",
            zero_division=0,
        )
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true_clean,
            y_pred,
            labels=self.all_labels,
            average="micro",
            zero_division=0,
        )

        # Average Precision (mAP) e ROC-AUC por classe (apenas 1 coluna por vez na RAM)
        ap_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)
        auc_per_class = np.full(self.num_classes, np.nan, dtype=np.float64)

        for c in self.all_labels:
            y_true_bin = (y_true_clean == c).astype(np.uint8)
            pos_count = int(np.sum(y_true_bin))
            class_name = self.id_to_name[c]

            if pos_count == 0:
                warnings_list.append(
                    f"Classe '{class_name}' (id={c}) possui support=0 em y_true; "
                    "excluída do cálculo de mAP e ROC-AUC macro."
                )
                continue

            col_scores = np.ascontiguousarray(y_prob_clean[:, c], dtype=np.float32)
            ap_per_class[c] = float(average_precision_score(y_true_bin, col_scores))

            if pos_count == num_samples:
                warnings_list.append(
                    f"Classe '{class_name}' (id={c}) representa 100% das amostras; "
                    "ROC-AUC indefinido."
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

        # Matriz de Confusão
        cm = confusion_matrix(y_true_clean, y_pred, labels=self.all_labels).astype(
            np.int64
        )
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm.astype(np.float64),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        )

        pred_counts = np.bincount(
            y_pred[y_pred >= 0], minlength=self.num_classes
        )

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
# Fachada Principal de Alto Nível
# ==============================================================================


def evaluate_yolo_classifier(
    model: Any,
    data_source: str | Path | Sequence[tuple[str | Path, int | str]],
    batch_size: int = 16,
    chunk_size: int = 512,
    imgsz: int | None = None,
    device: str | int | None = None,
    quantize: int | str | None = None,
    confidence_threshold: float = 0.0,
    strict_nan: bool = True,
    use_memmap: bool = False,
    memmap_dir: str | Path | None = None,
) -> EvaluationReport:
    """
    Avalia um modelo Ultralytics YOLO-cls com controle rígido de memória primária.
    """
    extractor = YOLOPredictionExtractor(model)
    y_true, y_prob = extractor.extract(
        data_source=data_source,
        batch_size=batch_size,
        chunk_size=chunk_size,
        imgsz=imgsz,
        device=device,
        quantize=quantize,
        use_memmap=use_memmap,
        memmap_dir=memmap_dir,
    )
    evaluator = ClassificationMetricsEvaluator(
        class_names=extractor.id_to_name,
        strict_nan=strict_nan,
    )
    report = evaluator.evaluate_arrays(
        y_true=y_true,
        y_prob=y_prob,
        confidence_threshold=confidence_threshold,
    )

    # Remove arquivo temporário caso memmap tenha sido utilizado
    if isinstance(y_prob, np.memmap) and y_prob.filename:
        memmap_path = y_prob.filename
        del y_prob
        _release_memory()
        try:
            os.remove(memmap_path)
        except OSError:
            pass

    return report


# ==============================================================================
# Suíte de Validação de Sanidade (Inclui Teste de Chunking + Memmap)
# ==============================================================================


if __name__ == "__main__":
    class _MockProbs:
        def __init__(self, vec: NDArray[np.float32]) -> None:
            if torch is not None:
                self.data = torch.tensor(vec, dtype=torch.float16)
            else:
                self.data = vec

    class _MockResult:
        def __init__(self, vec: NDArray[np.float32]) -> None:
            self.probs = _MockProbs(vec)

    class _MockYOLO26Cls:
        def __init__(self, probs_matrix: NDArray[np.float32]) -> None:
            self.task = "classify"
            self.names = {i: f"classe_{i:02d}" for i in range(28)}
            self._probs_matrix = probs_matrix
            self.calls_count = 0

        def predict(self, source: Sequence[str], stream: bool = True, **kwargs: Any):
            assert stream is True
            self.calls_count += 1
            for path_str in source:
                idx = int(path_str.rsplit("_", 1)[-1].split(".")[0])
                yield _MockResult(self._probs_matrix[idx])

    rng = np.random.default_rng(seed=42)
    N_SAMPLES = 2_000
    N_CLASSES = 28

    weights = np.array([1.0 / ((i + 1) ** 1.35) for i in range(N_CLASSES)])
    weights[27] = 0.0
    weights /= weights.sum()

    y_true_mock = rng.choice(N_CLASSES, size=N_SAMPLES, p=weights)
    logits = rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, N_CLASSES))
    for idx, true_cls in enumerate(y_true_mock):
        logits[idx, true_cls] += 2.8
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_mock = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(
        np.float32
    )

    mock_model = _MockYOLO26Cls(probs_mock)
    mock_manifest = [(f"/fake/val/img_{i}.jpg", int(y_true_mock[i])) for i in range(N_SAMPLES)]

    # Execução com chunk_size=256 e memmap ativado
    report = evaluate_yolo_classifier(
        model=mock_model,
        data_source=mock_manifest,
        batch_size=16,
        chunk_size=256,
        quantize=16,
        use_memmap=True,
    )
    assert mock_model.calls_count == 8, "Deveria ter dividido 2000 amostras em 8 chunks de 256!"
    print(report.format_summary())