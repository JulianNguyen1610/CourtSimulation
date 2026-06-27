"""Fine-tuned extractive QA reader for Vietnamese legal domain."""

from .finetune_reader import (
    FinetunedQAResult,
    LegalQADataset,
    LegalQAReader,
    ReaderConfig,
    finetune_reader,
    load_finetuned_reader,
)

__all__ = [
    "FinetunedQAResult",
    "LegalQADataset",
    "LegalQAReader",
    "ReaderConfig",
    "finetune_reader",
    "load_finetuned_reader",
]
