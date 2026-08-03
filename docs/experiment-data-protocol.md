# Experiment Data Protocol

Milestone 0 freezes ALQAC membership in `data/splits/alqac_v1.json`. The file records dataset SHA256, seed, ratio strategy, IDs and counts. It is a reviewed split manifest, **not** a claim that the historical 53-case test subset remains untouched.

Configuration precedence is YAML defaults, then explicit CLI values; omitted CLI options remain `None` and never replace YAML. The resolved, secret-free configuration is written to `config.json` and `run_manifest.json` with dataset, split and memory hashes.

Training/retrieval/memory construction may use train IDs only. Validation and test are read-only for memory. Existing memory JSON files remain historical artifacts. Reader evaluation requires `checkpoint_manifest.json` and rejects train/evaluation overlap.
