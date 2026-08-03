"""Single, validated experiment configuration and immutable split manifests."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SplitConfig(BaseModel):
    seed: int = 42
    strategy: Literal["ratios"] = "ratios"
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

    @model_validator(mode="after")
    def ratios_sum_to_one(self) -> "SplitConfig":
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")
        return self


class DatasetConfig(BaseModel):
    path: Path = Path("data/ALQAC.csv")
    split: SplitConfig = Field(default_factory=SplitConfig)
    split_manifest: Path = Path("data/splits/alqac_v1.json")


class MethodConfig(BaseModel):
    rounds: int = 1
    method: str = "both"
    limit: int = 10


class RetrievalConfig(BaseModel):
    legal_evidence_top_k: int = 5
    past_memory_top_k: int = 5
    method: str = "bm25_only"
    rough_top_n: int = 100
    reranker_model: str = "BAAI/bge-m3"


class MemoryConfig(BaseModel):
    path: Path = Path("memory-bank/train_only_memory.json")
    mode: Literal["off", "read_only", "read_update"] = "read_only"
    retrieval: Literal["lexical", "embedding"] = "lexical"
    max_entries_per_bucket: int = 1000
    embedding_model: str = "intfloat/multilingual-e5-large"


class PromptBudgetConfig(BaseModel):
    argument_max_tokens: int = 500
    max_context_chars: int | None = None
    max_evidence_docs: int | None = None
    max_evidence_chars: int | None = None
    max_history_turns: int | None = None
    max_history_chars: int | None = None


class EvaluationConfig(BaseModel):
    default_split: Literal["train", "validation", "test"] = "validation"
    enable_llm_evaluator: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = "vilqa_multi_agent_simple_baseline"
    seed: int = 42
    output_dir: Path = Path("outputs/vilqa_multi_agent_baseline")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    method: MethodConfig = Field(default_factory=MethodConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    prompt_budget: PromptBudgetConfig = Field(default_factory=PromptBudgetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        experiment = raw.get("experiment", {})
        dataset = raw.get("dataset", {})
        split = dataset.get("split", {}) if isinstance(dataset, dict) else {}
        debate = raw.get("debate", {})
        retrieval = raw.get("retrieval", {})
        memory = raw.get("memory", {})
        evaluation = raw.get("evaluation", {})
        batch = evaluation.get("batch", {}) if isinstance(evaluation, dict) else {}
        limits = debate.get("local_prompt_limits", {}) if isinstance(debate, dict) else {}
        return cls.model_validate({"name": experiment.get("name", cls.model_fields["name"].default), "seed": experiment.get("seed", 42), "output_dir": experiment.get("output_dir", "outputs/vilqa_multi_agent_baseline"), "dataset": {"path": dataset.get("path", "data/ALQAC.csv"), "split": {**split, "seed": experiment.get("seed", 42)}, "split_manifest": dataset.get("split_manifest", "data/splits/alqac_v1.json")}, "method": {"rounds": debate.get("rounds", 1), "method": batch.get("default_method", "both"), "limit": batch.get("default_limit", 10)}, "retrieval": retrieval, "memory": memory, "prompt_budget": {"argument_max_tokens": debate.get("argument_max_tokens", 500), **limits}, "evaluation": {"default_split": batch.get("default_split", "validation"), "enable_llm_evaluator": evaluation.get("enable_llm_evaluator", False)}})


class SplitManifest(BaseModel):
    dataset_path: str
    dataset_sha256: str
    seed: int
    split_strategy: str
    train_case_ids: list[str]
    validation_case_ids: list[str]
    test_case_ids: list[str]
    counts: dict[str, int]
    created_at: str

    @model_validator(mode="after")
    def validate_ids(self) -> "SplitManifest":
        groups = [self.train_case_ids, self.validation_case_ids, self.test_case_ids]
        flattened = [item for group in groups for item in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Split manifest contains overlapping case IDs")
        if self.counts != {"train": len(groups[0]), "validation": len(groups[1]), "test": len(groups[2])}:
            raise ValueError("Split manifest counts do not match case IDs")
        return self

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def new_split_manifest(dataset_path: str | Path, case_ids: list[str], split: SplitConfig) -> SplitManifest:
    import random
    ids = list(case_ids); random.Random(split.seed).shuffle(ids)
    total = len(ids); train_end = int(total * split.train_ratio); validation_end = train_end + int(total * split.validation_ratio)
    return SplitManifest(dataset_path=str(Path(dataset_path)), dataset_sha256=sha256_file(dataset_path), seed=split.seed, split_strategy="seeded_ratio", train_case_ids=ids[:train_end], validation_case_ids=ids[train_end:validation_end], test_case_ids=ids[validation_end:], counts={"train": train_end, "validation": validation_end-train_end, "test": total-validation_end}, created_at=datetime.now(timezone.utc).isoformat())
