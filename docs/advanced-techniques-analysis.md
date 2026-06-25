# Phân Tích Kỹ Thuật Nâng Cao — Legal Court Simulation

> Cập nhật: 2026-06-17  
> Nguồn: 6 paper đã review trong `document/`  
> Liên quan: [Tình trạng dự án](./project-status-and-overview.md) · [Checklist](./implementation-checklist.md)

---

## 1. Mục đích tài liệu

Tài liệu này tổng hợp các **kỹ thuật nâng cao** được báo cáo trong literature về legal multi-agent simulation, giải thích cơ chế, lợi ích kỳ vọng, rủi ro, và **mức độ áp dụng** vào codebase hiện tại của dự án.

---

## 2. Ma trận paper → kỹ thuật

| Paper | Venue / Năm | Task chính | Kỹ thuật cốt lõi | Relevance dự án |
|---|---|---|---|---|
| **AgenticSimLaw** | arXiv 2026 | High-stakes tabular decision (recidivism) | Private strategy + public debate; judge belief; explainability | Cao — debate loop Phase 1 |
| **AgentsCourt** | arXiv 2024 / EMNLP | Judicial decision-making end-to-end | Court debate + Legal RAG (BM25+BGE) + Agent-as-Judge | Rất cao — RAG + LJP + SimuCourt |
| **Courtroom-LLM** | COLING 2025 | Ambiguous text classification | Legal-inspired multi-LLM roles; similar-case retrieval | Cao — structured roles + precedent |
| **AgentCourt AdvEvol** | arXiv 2025 | Dynamic court simulation | Adversarial evolvable agents; 3-tier knowledge evolution | Rất cao — MemoryStore design |
| **MASER** | NAACL 2025 Findings | Intensive legal interaction | Multi-agent simulator driver; goal-directed questioning | Cao — Phase 3 protocol + defendant |
| **LegalSim** | arXiv 2025 | AI safety in legal systems | Procedural exploit discovery; rule-based MARL red-team | Trung bình — Phase 4 safety |

---

## 3. Phân tích theo khối pipeline

### 3.1 Input & chống leakage

#### Kỹ thuật

| Kỹ thuật | Paper | Mô tả |
|---|---|---|
| Agent-visible view tách khỏi gold label | AgenticSimLaw, AgentsCourt | Agent chỉ thấy facts/question; không thấy answer/judgment GT |
| Structured case schema | AgentsCourt, MASER | Facts, evidence, testimonies, applicable law tách field |
| Dataset adapter đa phase | — | Cùng case map sang QA profile (P1) và CourtCase (P3) |

#### Lợi ích

- Tránh data leakage khi train/eval prompt.
- Cho phép so sánh công bằng giữa direct vs debate vs courtroom.

#### Rủi ro nếu bỏ qua

- Metric inflated; kết luận nghiên cứu không hợp lệ.

#### Trạng thái dự án

**Đã có:** `CaseProfile.agent_view()`, `CourtCase`, `to_case_profile()` adapter.

---

### 3.2 Retrieve Legal Evidence (RAG pháp lý)

#### Kỹ thuật A — Two-stage retrieval (AgentsCourt)

```text
Query → BM25 rough top-N (100) → Semantic rerank top-K (5) → Agents
```

- **BM25**: nhanh, lexical, phù hợp thuật ngữ pháp lý VN/EN.
- **Semantic rerank** (BGE-m3, multilingual-e5): sửa lỗi BM25 miss paraphrase.
- **Rough top-N lớn**: recall cao trước khi rerank chính xác.

**Kỳ vọng:** +F1/legal_accuracy khi context dài hoặc câu hỏi cần điều luật ngoài ALQAC train.

**Rủi ro:** Latency và VRAM khi load sentence-transformers; rerank sai có thể đưa evidence nhiễu.

**Trạng thái:** BM25 + `SemanticReranker` có; default `bm25_only`; ablation `bm25_rerank` chưa execute đầy đủ.

#### Kỹ thuật B — External legal corpus (AgentsCourt)

- Index **UTS_VLC** (Bộ luật VN) thay vì chỉ train contexts ALQAC.
- Metadata: `article_id`, `law_name`, `source_type` để judge cite và audit.

**Kỳ vọng:** Giảm hallucination điều luật; cải thiện legal_accuracy rubric.

**Rủi ro:** Corpus outdated; article không khớp ngữ cảnh câu hỏi ALQAC.

**Trạng thái:** Loader UTS_VLC có (`--include-uts-vlc`); chưa báo cáo metric so sánh.

#### Kỹ thuật C — Similar-case retrieval (Courtroom-LLM)

- Retrieve **án tương tự / precedent** từ memory hoặc case bank.
- Dùng làm evidence bổ sung cho judge và defense.

**Kỳ vọng:** Mạnh với case mơ hồ (borderline); giảm variance giữa runs.

**Rủi ro:** Precedent sai domain; overfit pattern case cũ.

**Trạng thái:** **Chưa implement** (chỉ có case bucket trong memory).

---

### 3.3 Retrieve Past Memory (Knowledge Evolution)

#### Kỹ thuật (AgentCourt AdvEvol + MASER)

**Ba tầng memory:**

| Bucket | Nội dung | Ví dụ |
|---|---|---|
| `regulations` | Điều luật, quy tắc rút ra | "Điều 173 BLHS — trộm cắp tài sản" |
| `experiences` | Chiến lược tranh luận hiệu quả | "Opponent challenge thiếu chứng cứ vật chất" |
| `cases` | Tóm tắt case + outcome | "vilqa-42: phạt tù 07 năm — chiếm đoạt di vật" |

**Cơ chế vận hành:**

1. **Query memory** trước debate (lexical overlap hoặc embedding).
2. **Reflection prompt** sau debate — distill insight, không append raw transcript.
3. **Dedup + cap** (`max_entries_per_bucket`) tránh memory drift.
4. **Mode ablation:** `off` / `read_only` / `read_update`.

**Kỳ vọng:**

- Case sau học từ case trước (cross-case transfer).
- Prosecutor/Defense có "kinh nghiệm" tái sử dụng.

**Rủi ro:**

- Noisy memory làm tệ hơn baseline (negative transfer).
- Memory chứa gold answer leakage nếu reflection không kiểm soát.

**Trạng thái:** `MemoryStore` đầy đủ tính năng; default `read_only`; ablation chưa có kết quả công bố.

---

### 3.4 Debate Agents

#### Kỹ thuật A — Private strategy → Public utterance (AgenticSimLaw)

Mỗi lượt agent:

1. LLM call 1: sinh **private strategy** (không public cho đối phương).
2. LLM call 2: sinh **public argument** dựa trên strategy.

**Lợi ích:** Tách planning khỏi rhetoric; giảm argument vội vàng; dễ log audit nội bộ.

**Chi phí:** 2× API calls mỗi turn.

**Trạng thái:** **Đã có** trong `DebateAgent.generate_argument()`.

#### Kỹ thuật B — Role-conditioned adversarial prompts (AgentsCourt, Courtroom-LLM)

- Proponent/Prosecutor: buộc tội, nhấn tình tiết tăng nặng.
- Opponent/Defense: gỡ tội, giảm nhẹ, chỉ kẽ hở chứng cứ.
- Temperature cao hơn cho debaters, thấp hơn cho judge.

**Trạng thái:** Prompt Phase 1 + Phase 3 courtroom riêng; role LLM config trong YAML.

#### Kỹ thuật C — Closing statements (AgenticSimLaw, AgentsCourt)

Sau n vòng debate, mỗi bên có **lời kết** trước verdict.

**Kỳ vọng:** Judge có tóm tắt tranh luận rõ ràng; cải thiện verdict consistency.

**Trạng thái:** **Đã có** (`include_closing_statements`, courtroom closing phase).

#### Kỹ thuật D — Courtroom protocol 3 giai đoạn (MASER, AgentsCourt)

```text
Opening → Debate (n rounds) → Judgment (closing + deliberation + ruling)
```

Turn order cứng: Judge mở phiên → Prosecutor cáo trạng → Defendant khai → Defense → debate rounds → closing → deliberation → ruling.

**Trạng thái:** **Đã có** `CourtroomProtocol` + `CourtroomSession`.

#### Kỹ thuật E — Goal-directed questioning (MASER)

Luật sư hỏi có chủ đích để thu thập fact còn thiếu (intensive legal interaction).

**Trạng thái:** Một phần — `enable_judge_question` có; chưa có prosecutor/defense active questioning loop riêng.

#### Kỹ thuật F — Prompt compaction (practical)

Cắt transcript/evidence khi vượt context window (`prompt_compact.py`, `max_context_chars`).

**Trạng thái:** **Đã có** utils; courtroom YAML có giới hạn chars/turns.

---

### 3.5 Judge Agent

#### Kỹ thuật A — Belief tracking từng round (AgenticSimLaw, AgentsCourt)

Sau mỗi round: `{prediction, confidence, reasoning}` — lịch sử belief feed vào round sau và verdict cuối.

**Trạng thái:** **Đã có** `JudgeAgent.update_belief()`.

#### Kỹ thuật B — Structured JSON verdict + robust parsing

- Prompt yêu cầu JSON thuần.
- Fallback khi parse fail; retry 1 lần với LLM thật.
- Metric `fallback_rate` trong batch output.

**Trạng thái:** **Đã có**; cần theo dõi fallback < 5% trên LLM ổn định.

#### Kỹ thuật C — Early stopping theo confidence (AgenticSimLaw)

Dừng debate sớm nếu `confidence >= threshold` — tiết kiệm token.

**Trạng thái:** **Đã có** (`early_stop_confidence`); default null (chưa tune).

#### Kỹ thuật D — Optional judge clarification (Courtroom-LLM)

Judge hỏi 1 câu làm rõ trước closing nếu tranh luận mơ hồ.

**Trạng thái:** **Đã có** flag `enable_judge_question`; default off.

#### Kỹ thuật E — Dual verdict schema (dự án)

- Phase 1: QA span (`answer`, `prediction`).
- Phase 3: LJP structured (`charge`, `article`, `sentence`, `reasoning`, `cited_evidence_ids`).

**Trạng thái:** **Đã có** `render_ljp_verdict()` + QA backward-compat view.

#### Kỹ thuật F — Answer postprocess cho extractive QA

Rút span ngắn từ output dài (năm, tháng, tuổi, đồng) — quan trọng cho ViLQA EM.

**Trạng thái:** **Đã có** `shorten_legal_answer()`; áp dụng đồng đều mọi baseline.

---

### 3.6 Evaluator

#### Kỹ thuật A — Automated metrics có gold

| Task | Metrics |
|---|---|
| ViLQA QA | Exact Match, token F1 |
| LJP | Charge accuracy, article accuracy, sentence MAE/RMSE/bucket |

**Trạng thái:** **Đã có** `ViLQAEvaluator`, `LJPEvaluator`.

#### Kỹ thuật B — LLM-as-judge rubric (AgentsCourt)

Đánh giá **không dùng gold answer**:

- `legal_accuracy`
- `argument_quality`
- `logical_consistency`

**Trạng thái:** `EvaluatorAgent` có; `--enable-llm-evaluator` optional; chưa batch mặc định.

#### Kỹ thuật C — Citation & hallucination benchmarks

- **LegalCiteBench**: citation retrieval, error detection.
- **LegalHalBench**: 5 loại hallucination pháp lý.

**Trạng thái:** **Chưa implement**; hooks trong `LJPEvaluator` có placeholder.

#### Kỹ thuật D — Human evaluation

Subset human rubric cho realism/fairness courtroom.

**Trạng thái:** Config placeholder `human_eval_subset`; chưa wiring.

---

### 3.7 Update Memory

#### Kỹ thuật (AgentCourt AdvEvol)

Post-debate pipeline:

```text
DebateResult + EvalResult → memory_reflection.txt → LLM distill → append 3 buckets → save JSON
```

**Reflection prompt** tách:

- regulation insights
- debate strategy lessons
- case summary (không copy gold answer verbatim vào agent context case mới)

**Trạng thái:** **Đã có** `update_from_debate()` + reflection; cần ablation `memory_update_on`.

---

### 3.8 Baselines & ablation (phương pháp khoa học)

#### Baselines cần so sánh (AgentsCourt + best practice)

| Baseline | Mục đích |
|---|---|
| Direct LLM | Single-shot lower bound có LLM |
| CoT LLM | Single-agent reasoning |
| Vanilla Debate | Debate không judge/protocol |
| Extractive QA reader | Strong non-LLM floor |
| BM25 + reader | RAG floor không debate |
| Structured debate (full system) | Proposed method |

#### Ablation biến kiểm soát

| Biến | Giá trị |
|---|---|
| Retrieval | off / bm25_only / bm25_rerank |
| Memory | off / read_only / read_update |
| Rounds | 1 / 3 / 5 |
| Judge | off (vanilla) / on (structured) |
| Closing | on / off |
| Roles | proponent-opponent / prosecutor-defense (P3) |

**Trạng thái:** Plan + script có; execution chưa đầy đủ.

---

### 3.9 An toàn & red-teaming (LegalSim)

#### Kỹ thuật

- Mô phỏng tố tụng rule-based.
- RL/MARL agents tìm **procedural exploits**: kéo dài phiên, cost inflation, calendar pressure.
- Mục tiêu: phát hiện lỗ hổng protocol trước khi deploy nghiên cứu mở rộng.

#### Áp dụng dự án

- Audit log turn order violations.
- Stress test: agent có thể skip phase, lặp vô hạn, cite evidence ngoài retrieved set?
- Không ưu tiên Phase 1; **Phase 4** sau khi courtroom ổn định.

**Trạng thái:** **Chưa bắt đầu**.

---

## 4. Bảng tổng hợp: kỹ thuật × mức triển khai

| # | Kỹ thuật | Paper | Priority | Code | Eval |
|---|---|---|---|---|---|
| 1 | Agent view anti-leakage | AgenticSimLaw | P0 | ✅ | ✅ test |
| 2 | Private → public agent | AgenticSimLaw | P0 | ✅ | ✅ |
| 3 | Judge belief tracking | AgentsCourt | P0 | ✅ | ✅ |
| 4 | JSON verdict + fallback | AgentsCourt | P0 | ✅ | ✅ metric |
| 5 | BM25 retrieval | AgentsCourt | P0 | ✅ | 🟡 |
| 6 | Semantic rerank | AgentsCourt | P1 | 🟡 | ⬜ ablation |
| 7 | UTS_VLC corpus | AgentsCourt | P1 | 🟡 | ⬜ |
| 8 | Memory 3-tier | AdvEvol | P1 | ✅ | ⬜ ablation |
| 9 | Memory reflection | AdvEvol | P1 | ✅ | ⬜ |
| 10 | Closing statements | AgenticSimLaw | P1 | ✅ | 🟡 ablation |
| 11 | Early stop confidence | AgenticSimLaw | P2 | ✅ | ⬜ tune |
| 12 | Judge follow-up Q | Courtroom-LLM | P2 | ✅ | ⬜ ablation |
| 13 | Similar-case retrieval | Courtroom-LLM | P2 | ⬜ | ⬜ |
| 14 | Courtroom 3-phase protocol | MASER | P1 | ✅ | 🟡 smoke |
| 15 | Defendant testimony | MASER | P1 | ✅ | 🟡 smoke |
| 16 | LJP structured verdict | AgentsCourt | P1 | ✅ | 🟡 pilot |
| 17 | LLM rubric evaluator | AgentsCourt | P2 | ✅ | ⬜ optional |
| 18 | Citation validity | LegalCiteBench | P3 | ⬜ | ⬜ |
| 19 | Hallucination check | LegalHalBench | P3 | ⬜ | ⬜ |
| 20 | Procedural red-team | LegalSim | P3 | ⬜ | ⬜ |
| 21 | Ablation matrix | Best practice | P1 | ✅ script | ⬜ execute |
| 22 | Answer postprocess QA | Practical | P1 | ✅ | 🟡 re-run |

**Chú thích:** ✅ hoàn thành · 🟡 một phần · ⬜ chưa · P0/P1/P2/P3 = mức ưu tiên triển khai

---

## 5. Khuyến nghị ưu tiên theo impact / effort

### Quick wins (effort thấp, impact cao)

1. Chạy ablation `rounds_1` vs `rounds_3` trên validation (đã có script).
2. Bật `--enable-llm-evaluator` cho subset 20 case — bổ sung rubric ngoài EM/F1.
3. Re-run validation sau answer postprocess với qwen3.5 hoặc Gemini ổn định.

### Medium effort

4. Execute `bm25_plus_rerank` và `include-uts-vlc` ablations.
5. Execute memory ablations off/read_only/read_update.
6. Courtroom pilot Gemini trên `case_01_theft.json` + lưu transcript.

### High effort / research value

7. Similar-case retrieval + cross-case memory eval.
8. Batch courtroom runner + SimuCourt integration.
9. LegalSim-style protocol audit suite.

---

## 6. Giả định & failure modes cần theo dõi

| Giả định | Failure mode | Cách verify |
|---|---|---|
| Debate cải thiện reasoning | Over-extraction, paraphrase | Error analysis + EM/F1 |
| Memory giúp case sau | Noisy negative transfer | Ordered batch ablation |
| Rerank cải thiện evidence | Reranker đưa article sai | Legal accuracy rubric |
| Judge JSON ổn định | fallback_rate cao | metrics.json |
| Courtroom realistic | Agents lặp lại / skip phase | Protocol tests + human eval |
| Gemini/local đủ mạnh | Quota, token cut | Provider logging per method |

---

## 7. Tài liệu tham khảo

| Paper | Link |
|---|---|
| AgenticSimLaw | arXiv:2601.21936 |
| AgentsCourt | [arXiv:2403.02959](https://arxiv.org/abs/2403.02959) |
| Courtroom-LLM | COLING 2025 |
| AgentCourt AdvEvol | [arXiv:2408.08089](https://arxiv.org/abs/2408.08089) |
| MASER | [NAACL 2025 Findings](https://aclanthology.org/2025.findings-naacl.395/) |
| LegalSim | [arXiv:2510.03405](https://arxiv.org/abs/2510.03405) |

Dataset & benchmark bổ sung: [`legal-datasets-collection.md`](./legal-datasets-collection.md).
