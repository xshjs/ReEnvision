from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from policy_evaluation.defi_memory_models import cosine, iter_jsonl

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


@dataclass
class MemoryCase:
    memory_id: str
    task: str
    source: str
    future_feature: List[float]
    outcome_feature: List[float]
    mismatch_type: str
    success: bool
    correction_direction: str
    recoverability: str
    reflection_text: str
    repair_result: str = "unknown"
    timestamp: Optional[str] = None
    row_id: Optional[int] = None
    sequence_index: Optional[int] = None
    subtask_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def future_np(self) -> np.ndarray:
        return np.asarray(self.future_feature, dtype=np.float32)

    def outcome_np(self) -> np.ndarray:
        return np.asarray(self.outcome_feature, dtype=np.float32)


@dataclass
class MemoryCandidate:
    memory_id: str
    similarity_score: float
    mismatch_type: str
    success: bool
    correction_direction: str
    recoverability: str
    repair_result: str
    source: str
    row_id: Optional[int]
    case: MemoryCase


@dataclass
class RetrievedMemorySet:
    recent_candidates: List[MemoryCandidate]
    key_candidates: List[MemoryCandidate]

    def all_candidates(self) -> List[MemoryCandidate]:
        return list(self.recent_candidates) + list(self.key_candidates)

    def to_prompt_payload(self) -> Dict[str, Any]:
        def pack(items: Sequence[MemoryCandidate]) -> List[Dict[str, Any]]:
            rows = []
            for item in items:
                state_semantics = item.case.metadata.get("state_semantics", {})
                diagnostic_semantics = item.case.metadata.get("diagnostic_semantics", {})
                rows.append(
                    {
                        "memory_id": item.memory_id,
                        "similarity_score": round(float(item.similarity_score), 4),
                        "task": item.case.task,
                        "state_semantics": state_semantics,
                        "mismatch_type": item.mismatch_type,
                        "success": bool(item.success),
                        "correction_direction": item.correction_direction,
                        "recoverability": item.recoverability,
                        "repair_result": item.repair_result,
                        "diagnostic_semantics": diagnostic_semantics,
                        "source": item.source,
                        "row_id": item.row_id,
                    }
                )
            return rows

        return {
            "recent_candidates": pack(self.recent_candidates),
            "key_candidates": pack(self.key_candidates),
        }


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return rows / norms


class MemoryBank:
    def __init__(self, recent_capacity: int = 64) -> None:
        self.recent_capacity = int(recent_capacity)
        self.recent_cases: List[MemoryCase] = []
        self.key_cases: List[MemoryCase] = []
        self._recent_index = None
        self._key_index = None
        self._recent_mat: Optional[np.ndarray] = None
        self._key_mat: Optional[np.ndarray] = None

    @classmethod
    def from_jsonl(
        cls,
        key_memory_path: Optional[Path] = None,
        recent_memory_path: Optional[Path] = None,
        recent_capacity: int = 64,
    ) -> "MemoryBank":
        bank = cls(recent_capacity=recent_capacity)
        if key_memory_path is not None and key_memory_path.exists():
            bank.key_cases = [memory_case_from_row(row, source="key") for row in iter_jsonl(key_memory_path)]
        if recent_memory_path is not None and recent_memory_path.exists():
            bank.recent_cases = [memory_case_from_row(row, source="recent") for row in iter_jsonl(recent_memory_path)]
            bank.recent_cases = bank.recent_cases[-bank.recent_capacity :]
        bank.rebuild_indexes()
        return bank

    def rebuild_indexes(self) -> None:
        self._recent_mat = stack_features(self.recent_cases)
        self._key_mat = stack_features(self.key_cases)
        self._recent_index = build_faiss_index(self._recent_mat)
        self._key_index = build_faiss_index(self._key_mat)

    def add_recent(self, case: MemoryCase) -> None:
        self.recent_cases.append(case)
        self.recent_cases = self.recent_cases[-self.recent_capacity :]
        self.rebuild_indexes()

    def add_key(self, case: MemoryCase, merge_similarity: float = 0.995) -> None:
        merged = False
        for idx, existing in enumerate(self.key_cases):
            if existing.task != case.task:
                continue
            if cosine(existing.future_np(), case.future_np()) >= merge_similarity:
                self.key_cases[idx] = merge_memory_case(existing, case)
                merged = True
                break
        if not merged:
            self.key_cases.append(case)
        self.rebuild_indexes()

    def maybe_promote(
        self,
        case: MemoryCase,
        repeated_failure: bool = False,
        successful_repair: bool = False,
    ) -> None:
        if successful_repair or repeated_failure or (not case.success and case.mismatch_type not in {"none", "unknown"}):
            self.add_key(case)
        else:
            self.add_recent(case)

    def retrieve(
        self,
        future_feature: np.ndarray,
        task: str,
        topk_recent: int = 2,
        topk_key: int = 4,
        include_task_mismatch_fallback: bool = True,
    ) -> RetrievedMemorySet:
        future_feature = np.asarray(future_feature, dtype=np.float32).reshape(-1)
        recent_candidates = rank_cases(
            self.recent_cases,
            self._recent_mat,
            self._recent_index,
            future_feature,
            topk_recent,
            task=task,
            fallback=include_task_mismatch_fallback,
        )
        key_candidates = rank_cases(
            self.key_cases,
            self._key_mat,
            self._key_index,
            future_feature,
            topk_key,
            task=task,
            fallback=include_task_mismatch_fallback,
        )
        return RetrievedMemorySet(recent_candidates=recent_candidates, key_candidates=key_candidates)

    def save_jsonl(self, recent_path: Path, key_path: Path) -> None:
        recent_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        with recent_path.open("w") as handle:
            for case in self.recent_cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
        with key_path.open("w") as handle:
            for case in self.key_cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")


def build_faiss_index(mat: Optional[np.ndarray]):
    if faiss is None or mat is None or mat.size == 0:
        return None
    norm = _normalize_rows(mat.astype(np.float32))
    index = faiss.IndexFlatIP(norm.shape[1])
    index.add(norm)
    return index


def stack_features(cases: Sequence[MemoryCase]) -> Optional[np.ndarray]:
    if not cases:
        return None
    return np.stack([case.future_np() for case in cases], axis=0).astype(np.float32)


def rank_cases(
    cases: Sequence[MemoryCase],
    feature_mat: Optional[np.ndarray],
    index: Any,
    future_feature: np.ndarray,
    topk: int,
    task: str,
    fallback: bool,
) -> List[MemoryCandidate]:
    if not cases or topk <= 0:
        return []
    preferred_idx = [idx for idx, case in enumerate(cases) if case.task == task]
    if preferred_idx:
        candidates = [cases[idx] for idx in preferred_idx]
        mat = np.stack([feature_mat[idx] for idx in preferred_idx], axis=0).astype(np.float32) if feature_mat is not None else None
        return rank_cases(candidates, mat, build_faiss_index(mat), future_feature, topk, task="", fallback=False)
    if index is not None and feature_mat is not None:
        q = _normalize_rows(future_feature.reshape(1, -1).astype(np.float32))
        scores, ids = index.search(q, min(int(topk), len(cases)))
        out = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0:
                continue
            case = cases[idx]
            out.append(memory_candidate_from_case(case, float(score)))
        if out:
            return out
    ranked = sorted(
        ((cosine(future_feature, case.future_np()), case) for case in cases),
        key=lambda item: item[0],
        reverse=True,
    )
    out = [memory_candidate_from_case(case, score) for score, case in ranked[:topk]]
    if out or not fallback:
        return out
    mismatch_cases = [case for case in cases if case.mismatch_type not in {"none", "unknown"}]
    ranked = sorted(
        ((cosine(future_feature, case.future_np()), case) for case in mismatch_cases),
        key=lambda item: item[0],
        reverse=True,
    )
    return [memory_candidate_from_case(case, score) for score, case in ranked[:topk]]


def memory_candidate_from_case(case: MemoryCase, similarity_score: float) -> MemoryCandidate:
    return MemoryCandidate(
        memory_id=case.memory_id,
        similarity_score=float(similarity_score),
        mismatch_type=case.mismatch_type,
        success=bool(case.success),
        correction_direction=case.correction_direction,
        recoverability=case.recoverability,
        repair_result=case.repair_result,
        source=case.source,
        row_id=case.row_id,
        case=case,
    )


def memory_case_from_row(row: Dict[str, Any], source: str) -> MemoryCase:
    memory_id = str(row.get("memory_id") or f"{source.upper()}_{int(row.get('row_id', 0))}")
    future_feature = row.get("future_feature", row.get("future_features", row.get("defi_future_feature", [])))
    outcome_feature = row.get("outcome_feature", row.get("repair_delta", row.get("outcome_features", [])))
    return MemoryCase(
        memory_id=memory_id,
        task=str(row.get("task", "unknown")),
        source=source,
        future_feature=list(np.asarray(future_feature, dtype=np.float32).reshape(-1)),
        outcome_feature=list(np.asarray(outcome_feature, dtype=np.float32).reshape(-1)),
        mismatch_type=str(row.get("mismatch_type", "unknown")),
        success=bool(row.get("success", row.get("outcome", {}).get("success", False))),
        correction_direction=str(row.get("correction_direction", "maintain")),
        recoverability=str(row.get("recoverability", "uncertain")),
        reflection_text=str(row.get("reflection_text", "")),
        repair_result=str(row.get("repair_result", "unknown")),
        timestamp=row.get("timestamp"),
        row_id=int(row["row_id"]) if row.get("row_id") is not None else None,
        sequence_index=int(row["sequence_index"]) if row.get("sequence_index") is not None else None,
        subtask_index=int(row["subtask_index"]) if row.get("subtask_index") is not None else None,
        metadata={
            **dict(row.get("metadata", {})),
            "state_semantics": dict(row.get("state_semantics", {})),
            "diagnostic_semantics": dict(row.get("diagnostic_semantics", {})),
        },
    )


def merge_memory_case(old: MemoryCase, new: MemoryCase) -> MemoryCase:
    future_feature = 0.5 * (old.future_np() + new.future_np())
    outcome_feature = 0.5 * (old.outcome_np() + new.outcome_np())
    merged = MemoryCase(
        memory_id=old.memory_id,
        task=old.task,
        source="key",
        future_feature=future_feature.astype(np.float32).tolist(),
        outcome_feature=outcome_feature.astype(np.float32).tolist(),
        mismatch_type=new.mismatch_type if new.mismatch_type not in {"none", "unknown"} else old.mismatch_type,
        success=bool(new.success),
        correction_direction=new.correction_direction or old.correction_direction,
        recoverability=new.recoverability or old.recoverability,
        reflection_text=new.reflection_text or old.reflection_text,
        repair_result=new.repair_result if new.repair_result != "unknown" else old.repair_result,
        timestamp=new.timestamp or old.timestamp,
        row_id=new.row_id if new.row_id is not None else old.row_id,
        sequence_index=new.sequence_index if new.sequence_index is not None else old.sequence_index,
        subtask_index=new.subtask_index if new.subtask_index is not None else old.subtask_index,
        metadata={**old.metadata, **new.metadata},
    )
    return merged
