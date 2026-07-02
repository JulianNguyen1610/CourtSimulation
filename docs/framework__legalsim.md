# SE-NESMAD-EQA v3.1

## Span-Centric Self-Evolving Neuro-Symbolic Multi-Agent Framework for Extractive Legal QA

> Phiên bản: v3.1 — DPO-integrated profile  
> Phạm vi chính: Phase 1 — ViLQA / ALQAC Extractive Legal Question Answering  
> Ngôn ngữ ưu tiên: Tiếng Việt  
> Đối tượng trung tâm: `SpanCandidate`, không phải `Claim`  
> Mục tiêu: chọn đúng câu trả lời ngắn, nguyên văn, đúng boundary từ legal context.  
> Training profile chính: SFT → Span-level DPO trên Judge / Span Selector, dùng Qwen 3.5:9b local.

---

## Abstract

SE-NESMAD-EQA v3 là phiên bản span-centric của framework SE-NESMAD cũ, được thiết kế lại cho bài toán **extractive legal question answering**. Trong framework SE-NESMAD gốc, đối tượng trung tâm là `Claim`: hệ thống tạo claim, tranh luận claim, xác minh claim bằng logic engine, rồi đưa ra legal decision. Cấu trúc đó phù hợp với legal judgment prediction, courtroom simulation, hoặc claim verification.

Tuy nhiên, Phase 1 của dự án hiện tại là ViLQA / ALQAC extractive QA. Với bài toán này, câu hỏi không phải là “claim này đúng hay sai?”, mà là:

```text
Span nào trong legal context là câu trả lời tốt nhất?
```

Do đó, SE-NESMAD-EQA v3 thay đổi ontology gốc:

```text
Claim → SpanCandidate
Argument Graph → Span Candidate Graph
support / attack → contains / overlaps / refines / conflicts / dominates
proof verification → span verification
judge verdict → selected span with offsets
R_claim → R_span
```

Framework mới vẫn giữ tinh thần của SE-NESMAD cũ: retrieval, memory, multi-agent debate, evaluator độc lập, reflection, self-evolution, reward hierarchy, DPO và RLVR. Nhưng toàn bộ hệ thống được tái định vị quanh **candidate answer span** thay vì logical claim.

Bản v3.1 bổ sung một vòng **training-time optimization bằng Span-Level Direct Preference Optimization (DPO)**. DPO không thay thế inference pipeline; nó sử dụng output của pipeline, EM/F1/boundary reward và error analysis để tạo cặp `(prompt, chosen, rejected)`, sau đó fine-tune policy của Judge / Span Selector. Với cấu hình dự án hiện tại, target model thực nghiệm là **Qwen 3.5:9b**. Nếu Qwen 3.5:9b đang chạy qua Ollama/API chỉ để inference, DPO cần một checkpoint local có thể fine-tune bằng Transformers/TRL/LoRA hoặc QLoRA.

---

# 1. Motivation

LLM có thể trả lời tốt các câu hỏi pháp lý bằng ngôn ngữ tự nhiên, nhưng với extractive QA, hệ thống cần thỏa mãn các ràng buộc chặt hơn:

- câu trả lời phải xuất hiện nguyên văn trong context;
- câu trả lời phải ngắn và đúng boundary;
- không được paraphrase;
- không được thêm prefix / suffix thừa;
- không được mở rộng thành cả câu nếu gold answer chỉ là một cụm ngắn;
- không được trả về định nghĩa đầy đủ khi câu hỏi chỉ yêu cầu thuật ngữ;
- không được trả cả range khi câu hỏi yêu cầu upper bound hoặc lower bound.

Trong các thí nghiệm Phase 1, lỗi phổ biến không phải là sai logic pháp lý sâu, mà là lỗi **span boundary**:

```text
Gold:      15 năm
Pred bad:  bị phạt tù từ 07 năm đến 15 năm
```

hoặc:

```text
Gold:      Chiếm hữu
Pred bad:  Chiếm hữu là việc chủ thể nắm giữ...
```

Vì vậy, nếu giữ framework claim-centric, các layer như Argument Graph, QBAF, Proof Derivation, Counterexample Search sẽ bị lệch mục tiêu. Hệ thống cần một framework mới tập trung vào:

```text
legal localization → span candidate generation → span comparison → boundary verification → final span selection
```

---

# 2. Design Principle

## 2.1 Central object shift

Framework cũ:

```text
Claim
  ↓
Argument Graph
  ↓
QBAF / proof verification
  ↓
Final legal decision
```

Framework mới:

```text
SpanCandidate
  ↓
Span Candidate Graph
  ↓
Span Verification Ladder
  ↓
Final extractive answer
```

## 2.2 What the system optimizes

Framework cũ tối ưu:

```text
truth of a claim
```

Framework mới tối ưu:

```text
quality of an extractive span
```

Trong Phase 1, “đúng” không chỉ là đúng về mặt ngữ nghĩa. Một đáp án có thể đúng về ý nhưng vẫn sai EM/F1 nếu boundary sai.

Ví dụ:

```text
Question: Mức phạt tù cao nhất là bao lâu?
Gold:     15 năm
Pred:     phạt tù từ 07 năm đến 15 năm
```

Pred có thể đúng về ngữ nghĩa rộng, nhưng không phải span tối ưu cho extractive QA.

---

# 3. Overall Architecture

```text
Input: Legal Context + Question
    ↓
Task & Answer-Type Parser
    - question_type
    - answer_type
    - expected granularity
    - legal issue / proposition hint
    ↓
Retrieval & Legal Localization
    - BM25
    - optional semantic rerank
    - optional external legal corpus
    - optional PYTHEN rule localization
    ↓
Span Candidate Generation
    - regex candidates
    - legal term candidates
    - number / date / money / duration candidates
    - sentence-window candidates
    - retrieved regulation snippets
    - LLM-proposed candidates
    ↓
Span Candidate Graph
    - contains
    - overlaps
    - boundary_refines
    - same_normalized_answer
    - conflicts
    - dominates
    ↓
Multi-Agent Boundary Debate
    - Proponent proposes best span
    - Opponent challenges boundary / relevance / sufficiency
    - Judge controls debate and selects span
    ↓
Span Verification Ladder
    L0 Extractiveness
    L1 Boundary Validity
    L2 Relevance
    L3 Sufficiency
    L4 Dominance Search
    ↓
Judge Span Selection
    - answer
    - source_doc_id
    - start_offset
    - end_offset
    - confidence
    ↓
Evaluation
    - Exact Match
    - token F1
    - character IoU
    - boundary distance
    - optional LLM-as-judge
    ↓
Reflection & Memory Update
    - span policy
    - boundary failure
    - answer-type lesson
    ↓
Training-Time Self-Evolution
    - Span reward from EM / F1 / character IoU / boundary overlap
    - Preference Pair Builder: prompt + chosen span verdict + rejected span verdict
    - Span-Level DPO for Judge / Span Selector
    - optional DPO for Proponent and Opponent
    - optional RLVR / GRPO after DPO baseline is stable
    ↓
Updated Agent Policies
    - updated Qwen 3.5:9b Judge adapter / checkpoint
    - validated Agent Policy Memory
    - replay on validation set before deployment
```

---

# 4. Input Contract

Phase 1 input gồm:

```json
{
  "case_id": "vilqa-331",
  "context": "... legal context ...",
  "question": "Mức phạt tù cao nhất là bao lâu?",
  "metadata": {
    "dataset": "ALQAC",
    "split": "validation",
    "language": "vi"
  }
}
```

Gold answer chỉ được dùng ở evaluation / training reward, không được đưa vào agent prompt.

Agent-visible view:

```json
{
  "case_id": "vilqa-331",
  "context": "... legal context ...",
  "question": "Mức phạt tù cao nhất là bao lâu?"
}
```

Gold-hidden invariant:

```text
No agent, judge, retriever, memory query, or evaluator prompt may receive gold answer during inference.
```

---

# 5. Task & Answer-Type Parser

## 5.1 Role

Task parser xác định loại câu hỏi và dạng span cần trích. Đây là layer quan trọng để giảm over-extraction.

Input:

```json
{
  "question": "Mức phạt tù cao nhất là bao lâu?"
}
```

Output:

```json
{
  "question_type": "maximum_penalty",
  "answer_type": "duration",
  "expected_granularity": "upper_bound_only",
  "legal_issue": "penalty_duration",
  "constraints": {
    "must_be_short": true,
    "must_be_verbatim": true,
    "prefer_numeric_unit": true
  }
}
```

## 5.2 Common answer types

| Answer type | Description | Example |
|---|---|---|
| `duration` | thời hạn, số năm, số tháng, số ngày | `15 năm`, `01 tháng` |
| `money` | số tiền, mức phạt tiền | `100.000.000 đồng` |
| `date` | ngày, tháng, năm | `01/01/2025` |
| `legal_term` | thuật ngữ pháp lý | `Chiếm hữu` |
| `legal_subject` | chủ thể pháp lý | `người sử dụng lao động` |
| `condition_span` | điều kiện áp dụng | `khi có yêu cầu của bên mua` |
| `article_reference` | điều / khoản / điểm | `Điều 173` |
| `penalty_range` | khoảng hình phạt | `từ 07 năm đến 15 năm` |
| `upper_bound` | mức cao nhất | `15 năm` |
| `lower_bound` | mức thấp nhất | `07 năm` |

## 5.3 Expected granularity

`expected_granularity` giúp judge chọn boundary:

| Granularity | Meaning |
|---|---|
| `term_only` | chỉ lấy thuật ngữ |
| `number_unit_only` | chỉ lấy số + đơn vị |
| `upper_bound_only` | chỉ lấy cận trên |
| `lower_bound_only` | chỉ lấy cận dưới |
| `full_range` | lấy cả khoảng |
| `condition_phrase` | lấy cụm điều kiện |
| `sentence_level` | cần cả câu |
| `clause_level` | cần một mệnh đề |

---

# 6. Retrieval & Legal Localization Layer

## 6.1 Purpose

Retrieval không trực tiếp trả lời câu hỏi. Retrieval chỉ tìm vùng văn bản có khả năng chứa đáp án.

```text
Question
  ↓
BM25 / rerank
  ↓
retrieved context windows / regulations
  ↓
candidate regions
```

Output của retrieval:

```json
{
  "doc_id": "ctx_vilqa_331",
  "text": "Người phạm tội ... bị phạt tù từ 07 năm đến 15 năm.",
  "score": 12.83,
  "source_type": "given_context",
  "metadata": {
    "law_name": "Bộ luật Hình sự",
    "article": "Điều 173"
  }
}
```

## 6.2 Retrieval modes

| Mode | Description |
|---|---|
| `off` | chỉ dùng context gốc |
| `bm25_only` | lexical retrieval |
| `bm25_rerank` | BM25 rough top-N + semantic rerank top-K |
| `external_corpus` | thêm UTS_VLC hoặc corpus luật ngoài |
| `hybrid` | context gốc + legal corpus + memory |

## 6.3 Legal localization

Legal localization xác định đoạn luật hoặc vùng context liên quan trước khi trích span.

```text
Question
  ↓
legal issue detection
  ↓
applicable rule / article localization
  ↓
text region selection
  ↓
span candidate generation
```

---

# 7. PYTHEN as Localization Backend

## 7.1 Repositioning PYTHEN

Trong framework cũ, PYTHEN được mô tả như backend xác định proposition pháp lý có đúng hay không, ví dụ:

```text
theft_established = true / false
```

Trong SE-NESMAD-EQA, PYTHEN không dùng để quyết định final answer. PYTHEN dùng để xác định:

```text
điều luật nào / rule nào / vùng văn bản nào có khả năng chứa answer
```

Nói cách khác:

```text
PYTHEN answers: where to extract from.
Span selector answers: what exactly to extract.
```

## 7.2 Proposition-to-span mapping

Cần thêm mapping từ proposition id sang document span.

```json
{
  "proposition_id": "theft_penalty_upper_bound",
  "rule_id": "BLHS_173_penalty",
  "source_doc_id": "BLHS_2015_Article_173",
  "text_region": {
    "start_offset": 120,
    "end_offset": 390
  },
  "answer_policy": "extract_upper_bound_only"
}
```

## 7.3 PYTHEN output for Phase 1

```json
{
  "localized_rule_ids": ["BLHS_173_penalty"],
  "source_doc_ids": ["BLHS_2015_Article_173"],
  "candidate_text_regions": [
    {
      "doc_id": "BLHS_2015_Article_173",
      "start_offset": 120,
      "end_offset": 390,
      "reason": "Question asks maximum penalty for theft."
    }
  ],
  "answer_policy": "upper_bound_only"
}
```

## 7.4 Important limitation

PYTHEN does not solve boundary selection. It may identify the relevant article or rule, but it cannot determine whether the final answer should be:

```text
15 năm
```

or:

```text
từ 07 năm đến 15 năm
```

That decision belongs to the Span Candidate Graph, Span Verification Ladder, and Judge Span Selector.

---

# 8. Span Candidate Generation

## 8.1 Purpose

Span Candidate Generation tạo ra nhiều ứng viên có thể là đáp án. Hệ thống không nên để LLM sinh answer tự do, vì dễ hallucinate hoặc paraphrase.

Candidate sources:

```text
1. Regex extractor
2. Legal term extractor
3. Number / money / duration / date extractor
4. Sentence-window extractor
5. Retrieved regulation snippets
6. LLM-proposed spans with offset validation
7. Postprocess-generated candidates
```

## 8.2 Candidate schema

```json
{
  "span_id": "S12",
  "text": "15 năm",
  "normalized_text": "15 năm",
  "source_doc_id": "ctx_vilqa_331",
  "source_type": "given_context",
  "start_offset": 452,
  "end_offset": 458,
  "source_sentence": "bị phạt tù từ 07 năm đến 15 năm",
  "answer_type": "duration",
  "generation_method": "upper_bound_extractor",
  "scores": {
    "extractiveness": 1.0,
    "boundary": 0.96,
    "relevance": 0.91,
    "sufficiency": 0.88,
    "minimality": 0.95
  },
  "status": "candidate"
}
```

## 8.3 Candidate statuses

| Status | Meaning |
|---|---|
| `candidate` | ứng viên ban đầu |
| `extractive_valid` | span có trong source |
| `boundary_valid` | boundary hợp lệ |
| `relevant` | trả lời đúng câu hỏi |
| `sufficient` | đủ thông tin |
| `dominated` | bị candidate khác tốt hơn |
| `selected` | được chọn cuối cùng |
| `rejected` | bị loại |

## 8.4 Example

Question:

```text
Mức phạt tù cao nhất là bao lâu?
```

Source sentence:

```text
Người phạm tội bị phạt tù từ 07 năm đến 15 năm.
```

Generated candidates:

```json
[
  {
    "span_id": "S1",
    "text": "bị phạt tù từ 07 năm đến 15 năm",
    "generation_method": "sentence_window"
  },
  {
    "span_id": "S2",
    "text": "07 năm đến 15 năm",
    "generation_method": "range_extractor"
  },
  {
    "span_id": "S3",
    "text": "15 năm",
    "generation_method": "upper_bound_extractor"
  }
]
```

For `expected_granularity = upper_bound_only`, `S3` should dominate `S1` and `S2`.

---

# 9. Span Candidate Graph

## 9.1 Purpose

Span Candidate Graph thay thế Argument Graph trong Phase 1. Nó biểu diễn quan hệ giữa các candidate answer spans, không phải quan hệ support / attack giữa logical claims.

## 9.2 Node schema

```json
{
  "node_id": "S3",
  "type": "span_candidate",
  "text": "15 năm",
  "doc_id": "ctx_vilqa_331",
  "start_offset": 452,
  "end_offset": 458,
  "answer_type": "duration"
}
```

## 9.3 Edge ontology

| Edge | Meaning | Example |
|---|---|---|
| `contains` | A chứa B | full sentence contains `15 năm` |
| `contained_by` | A nằm trong B | `15 năm` contained by full range |
| `overlaps` | A và B trùng một phần | two boundary variants |
| `boundary_refines` | B cắt gọn đúng hơn A | `15 năm` refines full range |
| `same_normalized_answer` | normalize giống nhau | `01 tháng` vs `1 tháng` |
| `conflicts` | hai span khác vùng / khác loại | money span vs duration span |
| `dominates` | A tốt hơn B theo policy | upper bound dominates range |
| `requires_context` | span quá ngắn cần câu nguồn | `họ` requires antecedent |

## 9.4 Edge example

```json
{
  "edges": [
    {
      "source": "S1",
      "target": "S3",
      "type": "contains"
    },
    {
      "source": "S3",
      "target": "S1",
      "type": "dominates",
      "reason": "Question asks upper bound only; S3 is shorter and sufficient."
    }
  ]
}
```

## 9.5 Candidate Dominance Score

QBAF dialectical strength is replaced by Candidate Dominance Score.

```text
DominanceScore(span) =
    relevance_score
  + boundary_score
  + sufficiency_score
  + minimality_score
  + retrieval_support
  - dominated_penalty
```

This is not logical truth strength. It is span selection quality.

---

# 10. Multi-Agent Boundary Debate

## 10.1 Principle

Debate trong Phase 1 phải bị ràng buộc bởi candidate graph. Agent không được tranh luận lan man và không được sinh answer không có offset.

```text
All debate turns must refer to span_id.
Any new answer proposed by an agent must be validated as a SpanCandidate with source offsets.
```

## 10.2 Proponent Agent

Responsibilities:

```text
- Propose the best span.
- Explain why it matches answer_type.
- Explain why its boundary is minimal but sufficient.
- Cite source_doc_id and offset.
- Optionally propose a new candidate, but only if extractive.
```

Output:

```json
{
  "selected_span_id": "S3",
  "answer": "15 năm",
  "reason": "Câu hỏi hỏi mức cao nhất, nên chọn upper bound thay vì cả range.",
  "new_candidates": []
}
```

## 10.3 Opponent Agent

Responsibilities:

```text
- Challenge over-extraction.
- Challenge under-extraction.
- Challenge wrong answer type.
- Challenge wrong source region.
- Find a better candidate that dominates the proposed span.
```

Output:

```json
{
  "attack_type": "over_extraction",
  "target_span_id": "S1",
  "alternative_span_id": "S3",
  "reason": "S1 chứa cả range, nhưng câu hỏi chỉ cần upper bound."
}
```

## 10.4 Judge Agent

Responsibilities:

```text
- Control debate.
- Track candidate scores.
- Apply Span Verification Ladder.
- Select final span with offsets.
- Reject non-extractive or boundary-invalid outputs.
```

Output:

```json
{
  "answer": "15 năm",
  "selected_span_id": "S3",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 452,
  "end_offset": 458,
  "source_sentence": "bị phạt tù từ 07 năm đến 15 năm",
  "confidence": 0.91,
  "answer_type": "duration",
  "selection_reason": "The question asks for the maximum duration; the upper-bound span dominates the full penalty range."
}
```

---

# 11. Debate Protocol

## 11.1 Default protocol

Phase 1 should default to one debate round.

```text
Round 0:
  - Parse question
  - Retrieve relevant context
  - Generate span candidates
  - Build span graph
  - Initial judge scoring

Round 1:
  - Proponent selects best span
  - Opponent challenges boundary / relevance / sufficiency
  - Judge selects final span or asks for one clarification
```

## 11.2 Adaptive extra rounds

Extra rounds should be used only when needed.

```text
Open extra round if:
- top-2 candidates overlap heavily;
- answer_type is uncertain;
- judge confidence < threshold;
- L4 dominance search finds unresolved alternative;
- candidate graph contains conflicting spans from different regions.
```

## 11.3 Stopping condition

```text
Stop if:
- selected span passes L0-L3;
- no dominant alternative found in L4;
- confidence >= threshold;
- max_round reached.
```

## 11.4 Anti-noise rule

Long debate can increase over-extraction. Therefore:

```text
The system should prefer candidate-space search over debate-space expansion.
```

For Phase 1, debate is not for generating more legal reasoning. Debate is for testing boundary quality.

---

# 12. Span Verification Ladder

Framework cũ dùng:

```text
L0 Syntax
L1 Consistency
L2 Rule Satisfaction
L3 Proof Derivation
L4 Counterexample Search
```

SE-NESMAD-EQA dùng:

```text
L0 Extractiveness
L1 Boundary Validity
L2 Relevance
L3 Sufficiency
L4 Dominance Search
```

## 12.1 L0 Extractiveness

Check whether the span appears verbatim in the source document.

Pass:

```text
answer = "15 năm"
source contains "15 năm"
```

Fail:

```text
answer = "mười lăm năm"
source contains "15 năm"
```

unless normalization policy explicitly allows this.

## 12.2 L1 Boundary Validity

Check whether start/end boundaries are minimal and do not include unnecessary prefix/suffix.

Fail example:

```text
Gold: 01 tháng
Pred: Sau 01 tháng
```

## 12.3 L2 Relevance

Check whether the candidate answers the question and matches answer type.

Fail example:

```text
Question asks duration.
Predicted span is money amount.
```

## 12.4 L3 Sufficiency

Check whether the span contains enough information.

Fail example:

```text
Question asks condition + deadline.
Predicted span contains only deadline.
```

## 12.5 L4 Dominance Search

Search top candidates for a better span.

A candidate dominates another if it is:

```text
- equally or more relevant;
- more minimal;
- boundary cleaner;
- sufficient;
- extractive;
- better aligned with expected granularity.
```

Example:

```text
S1 = "bị phạt tù từ 07 năm đến 15 năm"
S3 = "15 năm"
```

If question asks maximum duration, `S3` dominates `S1`.

## 12.6 Verification result schema

```json
{
  "span_id": "S3",
  "checks": {
    "L0_extractiveness": "pass",
    "L1_boundary": "pass",
    "L2_relevance": "pass",
    "L3_sufficiency": "pass",
    "L4_dominance": "pass"
  },
  "final_status": "selected"
}
```

---

# 13. Evaluator Agent

## 13.1 Role

Evaluator Agent vẫn giữ nguyên nguyên tắc độc lập của SE-NESMAD cũ, nhưng đổi tiêu chí từ claim correctness sang span quality.

Evaluator does not participate in debate. It only scores candidate spans.

## 13.2 Independence requirements

1. **Reward independence**: Evaluator reward must not derive from Judge verdict.
2. **Information independence**: Evaluator sees only question, span, source, and task profile; no debate history.
3. **Training independence**: Evaluator should be calibrated on a distribution emphasizing boundary errors, over-extraction, under-extraction, and answer-type mismatch.

## 13.3 Input schema

```json
{
  "question": "Mức phạt tù cao nhất là bao lâu?",
  "span_id": "S3",
  "span_text": "15 năm",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 452,
  "end_offset": 458,
  "source_sentence": "bị phạt tù từ 07 năm đến 15 năm",
  "answer_type": "duration",
  "expected_granularity": "upper_bound_only"
}
```

## 13.4 Output schema

```json
{
  "span_id": "S3",
  "precision_score": 0.98,
  "recall_score": 0.92,
  "boundary_score": 0.96,
  "relevance_score": 0.91,
  "minimality_score": 0.95,
  "uncertainty_flag": "low",
  "error_type": null
}
```

## 13.5 LLM-as-judge rubric

Existing rubric:

```text
legal_accuracy
argument_quality
logical_consistency
```

Additional Phase 1 rubric:

```text
extractiveness
boundary_quality
minimality
answer_type_match
sufficiency
```

---

# 14. Reward Architecture

## 14.1 Span reward

`R_claim` is replaced by `R_span`.

Inference-time score:

```text
Score_span =
    0.30 * relevance
  + 0.25 * boundary
  + 0.20 * sufficiency
  + 0.15 * minimality
  + 0.10 * retrieval_support
  - 0.40 * non_extractive_penalty
  - 0.30 * over_extraction_penalty
```

## 14.2 Training-time reward with gold span

For ALQAC / ViLQA, reward can be directly computed from gold answer.

```text
R_span =
  extractiveness_gate * (
      0.45 * exact_match
    + 0.35 * token_f1
    + 0.20 * char_iou
  )
  - 0.30 * length_excess_penalty
  - 0.20 * wrong_answer_type_penalty
```

Where:

```text
extractiveness_gate = 1 if predicted span appears in source context
extractiveness_gate = 0 if prediction is paraphrased or hallucinated
```

## 14.3 Proponent reward

```text
R_proponent =
    0.50 * selected_span_quality
  + 0.25 * evidence_grounding
  + 0.25 * boundary_reasoning
  - 0.40 * hallucinated_span
```

## 14.4 Opponent reward

```text
R_opponent =
    0.45 * valid_boundary_error_detection
  + 0.35 * better_span_proposed
  + 0.20 * concise_critique
  - 0.35 * false_attack
```

## 14.5 Judge reward

```text
R_judge =
    0.50 * final_span_quality
  + 0.25 * calibration
  + 0.25 * selection_explanation
  - 0.30 * ignored_dominant_candidate
```

## 14.6 Episode reward

```text
R_episode =
    0.45 * R_span
  + 0.25 * R_judge
  + 0.15 * R_proponent
  + 0.15 * R_opponent
```

---

# 15. Training-Time Optimization with Span-Level DPO

## 15.1 Position of DPO in SE-NESMAD-EQA

DPO belongs to the **training-time self-evolution loop**, not to the normal inference-time pipeline.

Inference-time pipeline:

```text
Context + Question
  → Retrieval / Localization
  → Span Candidate Generation
  → Span Candidate Graph
  → Multi-Agent Boundary Debate
  → Judge Span Selection
  → Final Extractive Answer
```

Training-time DPO loop:

```text
Episode output
  → deterministic span reward
  → preference pair construction
  → DPO fine-tuning
  → updated agent policy
  → validation replay
  → deploy or rollback
```

The purpose of DPO is to teach the model that, under the same prompt and candidate graph, one span verdict is preferable to another.

For Phase 1, DPO optimizes:

```text
Span selection quality, not claim truth.
```

Therefore, the DPO unit is:

```text
(prompt, chosen_span_verdict, rejected_span_verdict)
```

not:

```text
(prompt, verified_claim, refuted_claim)
```

This keeps the framework logically aligned with the span-centric ontology.

---

## 15.2 Why DPO fits Phase 1

Phase 1 has supervised extractive QA data. Each prediction can be scored by deterministic metrics:

- exact match;
- token F1;
- character overlap / character IoU;
- extractiveness;
- boundary distance;
- over-extraction penalty;
- under-extraction penalty;
- answer-type match.

This makes DPO natural because preference labels can be derived automatically:

```text
higher span reward → chosen
lower span reward  → rejected
```

The typical Phase 1 failure is not that the model cannot reason about law at all. The dominant failure is often boundary behavior:

```text
Gold:      15 năm
Bad pred:  bị phạt tù từ 07 năm đến 15 năm
```

or:

```text
Gold:      Chiếm hữu
Bad pred:  Chiếm hữu là việc chủ thể nắm giữ...
```

DPO can directly train the Judge / Span Selector to prefer the shorter, boundary-correct, sufficient answer span over the longer or paraphrased output.

---

## 15.3 Main DPO target: Judge / Span Selector

The first and most important DPO target is the Judge, specifically the Judge's final span-selection behavior.

The Judge receives:

```text
question
context
retrieved evidence
memory snippets
span candidates
span graph relations
proponent argument
opponent challenge
optional debate transcript
```

The Judge outputs:

```json
{
  "answer": "15 năm",
  "selected_span_id": "S2",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 432,
  "end_offset": 438,
  "confidence": 0.91,
  "reason": "Câu hỏi hỏi mức cao nhất nên chọn upper bound."
}
```

DPO trains the Judge to assign higher probability to the better verdict JSON.

Chosen example:

```json
{
  "answer": "15 năm",
  "selected_span_id": "S2",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 432,
  "end_offset": 438,
  "reason": "Câu hỏi hỏi mức cao nhất nên chỉ chọn upper bound."
}
```

Rejected example:

```json
{
  "answer": "bị phạt tù từ 07 năm đến 15 năm",
  "selected_span_id": "S1",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 410,
  "end_offset": 438,
  "reason": "Đây là toàn bộ range hình phạt."
}
```

The chosen and rejected outputs must use the **same schema as inference-time Judge output**. This prevents a train/inference mismatch.

---

## 15.4 Secondary DPO targets

### 15.4.1 Proponent DPO

After the Judge DPO baseline is stable, DPO can be applied to the Proponent.

Goal:

```text
Teach the Proponent to propose better initial spans.
```

Chosen:

```text
Proposes a short, extractive, answer-type-correct span with valid offset.
```

Rejected:

```text
Proposes a long sentence, paraphrase, wrong region, or wrong answer type.
```

This is useful when the Judge can only select among candidates already proposed by the system.

### 15.4.2 Opponent DPO

Opponent DPO is harder and should come later.

Goal:

```text
Teach the Opponent to identify real boundary and sufficiency errors.
```

Chosen critique:

```text
The selected span is over-extracted; S2 dominates S1 because the question asks only for the upper bound.
```

Rejected critique:

```text
Generic disagreement, unsupported attack, or criticism unrelated to boundary quality.
```

Opponent reward should be tied to whether the critique helps the Judge choose a better final span.

---

## 15.5 DPO mathematical objective

For each preference sample:

```text
x        = prompt
 y+      = chosen span verdict
 y-      = rejected span verdict
πθ       = trainable Qwen 3.5:9b policy
πref     = frozen reference model, usually the SFT checkpoint
β        = DPO temperature / KL strength
```

The DPO loss is:

```text
L_DPO_span(θ) = - log σ(
  β [
      log πθ(y+ | x) - log πref(y+ | x)
    - log πθ(y- | x) + log πref(y- | x)
    ]
)
```

Operationally:

```text
increase probability of the chosen JSON verdict
reduce probability of the rejected JSON verdict
keep the updated model close to the reference model
```

For this framework, the loss should be named:

```text
L_DPO_span
```

not:

```text
L_DPO_claim
```

because the preference object is a selected answer span.

---

## 15.6 Span reward used to build DPO pairs

DPO itself does not require a scalar reward during optimization, but the framework uses scalar span reward to construct preference pairs.

Recommended span reward:

```text
R_span =
    0.45 * exact_match
  + 0.35 * token_f1
  + 0.20 * char_iou
  - 0.30 * over_extraction_penalty
  - 0.25 * non_extractive_penalty
  - 0.20 * wrong_answer_type_penalty
```

Optional boundary-aware extension:

```text
R_span =
    0.40 * exact_match
  + 0.30 * token_f1
  + 0.15 * char_iou
  + 0.15 * boundary_score
  - 0.30 * over_extraction_penalty
  - 0.25 * under_extraction_penalty
  - 0.25 * non_extractive_penalty
```

Hard gates:

```text
if predicted answer is not extractive from context:
    apply non_extractive_penalty

if source_doc_id / offset is required but missing:
    reduce boundary_score

if answer_type is incompatible with question_type:
    apply wrong_answer_type_penalty
```

Pair construction rule:

```text
if R(candidate_a) - R(candidate_b) >= preference_min_delta:
    chosen   = candidate_a
    rejected = candidate_b
```

Default:

```text
preference_min_delta = 0.20
```

---

## 15.7 Preference pair sources

### 15.7.1 Candidate-level pairs

Generate multiple span candidates for the same case:

```text
regex candidate
LLM candidate
postprocessed candidate
reader candidate
vanilla debate output
judge-mediated output
```

Score all candidates and rank them.

Example:

```text
S1 = "bị phạt tù từ 07 năm đến 15 năm"  → lower reward
S2 = "15 năm"                           → higher reward
```

DPO pair:

```text
chosen   = S2 verdict
rejected = S1 verdict
```

### 15.7.2 Cross-method pairs

Use outputs from different system variants:

- Direct;
- CoT;
- Vanilla Debate;
- Fixed Debate;
- Judge-mediated Debate;
- Extractive Reader;
- BM25 + Reader;
- postprocessed variants.

For the same case:

```text
chosen   = method output with higher span reward
rejected = method output with lower span reward
```

### 15.7.3 Error-analysis pairs

Use known error patterns to create high-value pairs:

```text
chosen:   "01 tháng"
rejected: "Sau 01 tháng"
error:    prefix_error
```

```text
chosen:   "Chiếm hữu"
rejected: "Chiếm hữu là việc chủ thể nắm giữ..."
error:    definition_expansion
```

```text
chosen:   "100.000.000 đồng"
rejected: "phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng"
error:    over_extraction_range
```

These pairs directly target the boundary failures that reduce EM.

---

## 15.8 DPO data schema

Recommended JSONL path:

```text
data/dpo/span_judge_preferences.jsonl
```

Each line:

```json
{
  "id": "vilqa_331_judge_pair_001",
  "split": "train",
  "task": "extractive_legal_qa",
  "role": "judge_span_selector",
  "prompt": {
    "question": "Mức phạt tù cao nhất là bao lâu?",
    "context": "...",
    "retrieved_evidence": [
      {
        "doc_id": "ctx_331",
        "text": "..."
      }
    ],
    "memory": {
      "regulations": [],
      "experiences": [
        "Nếu câu hỏi hỏi mức cao nhất, ưu tiên upper bound thay vì toàn bộ range."
      ],
      "cases": []
    },
    "candidate_spans": [
      {
        "span_id": "S1",
        "text": "bị phạt tù từ 07 năm đến 15 năm",
        "start_offset": 410,
        "end_offset": 438,
        "answer_type": "duration"
      },
      {
        "span_id": "S2",
        "text": "15 năm",
        "start_offset": 432,
        "end_offset": 438,
        "answer_type": "duration"
      }
    ],
    "candidate_relations": [
      {
        "source": "S1",
        "target": "S2",
        "type": "contains"
      },
      {
        "source": "S2",
        "target": "S1",
        "type": "dominates",
        "reason": "upper_bound_only"
      }
    ]
  },
  "chosen": {
    "answer": "15 năm",
    "selected_span_id": "S2",
    "source_doc_id": "ctx_331",
    "start_offset": 432,
    "end_offset": 438,
    "reason": "Câu hỏi hỏi mức cao nhất nên chỉ chọn upper bound."
  },
  "rejected": {
    "answer": "bị phạt tù từ 07 năm đến 15 năm",
    "selected_span_id": "S1",
    "source_doc_id": "ctx_331",
    "start_offset": 410,
    "end_offset": 438,
    "reason": "Câu trả lời lấy cả range nên bị over-extraction."
  },
  "scores": {
    "chosen_em": 1.0,
    "chosen_f1": 1.0,
    "rejected_em": 0.0,
    "rejected_f1": 0.67,
    "reward_delta": 0.33
  },
  "error_type": "over_extraction"
}
```

For TRL-style training, the same record can be rendered into three strings:

```text
prompt   = render_prompt(record["prompt"])
chosen   = render_verdict(record["chosen"])
rejected = render_verdict(record["rejected"])
```

The rendered `chosen` and `rejected` strings should be valid JSON verdicts.

---

## 15.9 Qwen 3.5:9b DPO profile

The project currently uses `qwen3.5:9b` for local inference. For DPO, the key requirement is that the model must be available as a trainable checkpoint, not only as a black-box generation endpoint.

Important distinction:

```text
qwen3.5:9b via Ollama / API:
    usable for inference, batch prediction, candidate generation, evaluation runs
    not sufficient for DPO weight updates unless log-probs and weights are accessible

Qwen 3.5:9b as HuggingFace / local trainable checkpoint:
    usable for SFT, DPO, LoRA, QLoRA, validation, checkpoint deployment
```

Recommended practical setup:

```text
base model:       Qwen 3.5:9b instruct-style checkpoint
training method:  QLoRA + DPO
DPO target:       Judge / Span Selector first
reference model:  SFT checkpoint
inference model:  DPO adapter merged or loaded with base Qwen 3.5:9b
```

If a trainable Qwen 3.5:9b checkpoint is unavailable, the framework can still prepare DPO datasets and run offline evaluation, but actual DPO fine-tuning must wait until a trainable checkpoint is available.

---

## 15.10 Recommended Qwen 3.5:9b DPO config

```yaml
training:
  method: dpo
  target_agent: judge_span_selector
  base_model: qwen3.5:9b
  trainable_backend: transformers_trl
  adapter: qlora
  reference_model: sft_qwen3_5_9b_span_judge

sft:
  purpose: learn_json_span_verdict_format
  output_schema:
    - answer
    - selected_span_id
    - source_doc_id
    - start_offset
    - end_offset
    - confidence
    - reason

dpo:
  beta: 0.1
  learning_rate: 1.0e-6
  max_prompt_length: 2048
  max_length: 3072
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  num_train_epochs: 1
  preference_min_delta: 0.20
  label_smoothing: 0.0
  save_strategy: epoch
  eval_strategy: steps
  eval_steps: 100

reward:
  exact_match_weight: 0.45
  token_f1_weight: 0.35
  char_iou_weight: 0.20
  over_extraction_penalty: 0.30
  non_extractive_penalty: 0.25
  wrong_answer_type_penalty: 0.20

data:
  dpo_train_file: data/dpo/span_judge_preferences.train.jsonl
  dpo_valid_file: data/dpo/span_judge_preferences.valid.jsonl
  use_train_split_only_for_pair_construction: true
  exclude_test_split: true
  filter_ambiguous_gold: true
  min_reward_delta: 0.20

evaluation:
  metrics:
    - exact_match
    - token_f1
    - char_iou
    - extractiveness_rate
    - over_extraction_rate
    - under_extraction_rate
    - wrong_region_rate
    - json_valid_rate
    - fallback_rate
```

Tuning notes:

- If the model drifts away from JSON format, increase `beta` or add more SFT warmup.
- If the model barely changes after DPO, reduce `beta` slightly or increase preference quality.
- If answers become too short, increase sufficiency and under-extraction penalties.
- If answers remain too long, increase over-extraction penalty and add more boundary-error pairs.

---

## 15.11 Training procedure

Recommended sequence:

```text
Step 1: Run baseline systems on train split
Step 2: Save predictions, candidate spans, transcripts, metrics
Step 3: Compute span rewards for all outputs
Step 4: Build preference pairs with reward_delta >= threshold
Step 5: Filter noisy or ambiguous pairs
Step 6: SFT Qwen 3.5:9b on JSON span verdict format
Step 7: DPO fine-tune Judge / Span Selector
Step 8: Evaluate on validation split
Step 9: Compare against SFT and original qwen3.5:9b baseline
Step 10: Accept checkpoint only if validation improves and boundary errors decrease
Step 11: Freeze final config before one-shot test evaluation
```

A DPO checkpoint should not be accepted solely because DPO training loss decreases. It must improve task metrics:

```text
validation EM
validation token F1
extractiveness rate
JSON valid rate
boundary error reduction
over-extraction reduction
```

---

## 15.12 Preference filtering and anti-leakage rules

Preference data must be filtered before DPO.

Required filters:

```text
reward_delta >= 0.20
chosen answer is non-empty
chosen answer is extractive from the source context
chosen and rejected differ after normalization
source split is train or dedicated preference-train split
no test gold answer is used
ambiguous gold answers are excluded or manually reviewed
```

Anti-leakage policy:

```text
Do not store raw test gold answers in memory.
Do not build DPO pairs from test predictions using test labels before final reporting.
Do not inject case-specific gold spans into future prompts.
Store reusable span policies, not raw answer memorization.
```

Correct memory lesson:

```json
{
  "failure_type": "over_extraction_range",
  "lesson": "Nếu câu hỏi hỏi mức cao nhất, chọn upper bound thay vì toàn bộ range.",
  "applicable_answer_type": "duration",
  "validated": true
}
```

Incorrect memory lesson:

```json
{
  "case_id": "vilqa_331",
  "gold_answer": "15 năm"
}
```

---

## 15.13 DPO and Memory Graph

DPO interacts with memory in two ways.

First, memory can be part of the DPO prompt:

```text
question + context + retrieved evidence + relevant span-policy memory + candidate graph
```

Second, DPO results update Agent Policy Memory after validation:

```json
{
  "agent": "JudgeSpanSelector",
  "before": "Often selected full penalty ranges for maximum-duration questions.",
  "after": "Prefers upper-bound spans when question asks for maximum duration.",
  "validation_delta": {
    "exact_match": "+3.2 pp",
    "over_extraction_rate": "-18%"
  },
  "status": "validated"
}
```

DPO must not turn memory into answer leakage. Memory stores **generalizable span-selection policies**, not case-specific answer keys.

---

## 15.14 DPO and debate rounds

DPO should not be used to make debate longer. Phase 1 evidence suggests that more rounds can add noise and over-extraction. Therefore, DPO should teach the system to:

- select early when the best span is clear;
- stop when L0-L3 pass and L4 finds no dominant alternative;
- avoid unnecessary expansion after the correct span has been found;
- ask for another round only when candidate dominance is unresolved.

Possible preference pair for stopping behavior:

```json
{
  "chosen": {
    "action": "STOP_AND_SELECT",
    "answer": "15 năm"
  },
  "rejected": {
    "action": "CONTINUE_DEBATE_AND_EXPAND",
    "answer": "bị phạt tù từ 07 năm đến 15 năm"
  },
  "reason": "Continuing debate caused over-extraction."
}
```

This is still DPO over textual/action outputs, not PPO. PPO should be reserved for a later controller-training phase if the debate controller is formalized as a state/action/reward environment.

---

## 15.15 Evaluation after DPO

DPO evaluation must compare at least:

```text
Base qwen3.5:9b
SFT qwen3.5:9b Span Judge
DPO qwen3.5:9b Span Judge
DPO Judge + original Proponent/Opponent
DPO Judge + DPO Proponent, optional later
```

Required metrics:

| Metric | Purpose |
|---|---|
| Exact Match | Main extractive QA score |
| Token F1 | Partial overlap quality |
| Character IoU | Boundary-sensitive overlap |
| Extractiveness rate | Whether answer is copied from context |
| Over-extraction rate | Whether answer is too long |
| Under-extraction rate | Whether answer is too short |
| Wrong-region rate | Whether answer comes from wrong sentence/doc |
| JSON valid rate | Output contract reliability |
| Fallback rate | Judge parser stability |
| Token cost | Inference efficiency |

Acceptance condition:

```text
DPO checkpoint is accepted only if:
    validation EM/F1 improves or stays stable,
    over-extraction decreases,
    JSON valid rate does not degrade,
    fallback rate does not increase materially,
    no test leakage occurs.
```

---

## 15.16 Minimal implementation modules

Add or extend:

```text
src/training/span_reward.py
src/training/preference_builder.py
src/training/dpo_dataset_writer.py
src/training/dpo_renderers.py
src/training/dpo_train_qwen.py
```

Responsibilities:

```text
span_reward.py:
    compute EM, F1, char IoU, boundary score, penalties

preference_builder.py:
    rank outputs and create chosen/rejected pairs

dpo_dataset_writer.py:
    write JSONL files for train/validation DPO

dpo_renderers.py:
    convert structured prompt/chosen/rejected records into TRL-compatible strings

dpo_train_qwen.py:
    run SFT/DPO using Qwen 3.5:9b trainable checkpoint with LoRA/QLoRA
```

The DPO modules should not change inference behavior directly. They produce an updated model adapter/checkpoint that is later loaded by the Judge role.

---

## 15.17 Recommended implementation order

```text
1. Keep current qwen3.5:9b inference pipeline unchanged.
2. Add span reward and preference pair builder.
3. Generate DPO dataset from train split predictions.
4. SFT Qwen 3.5:9b on JSON span verdict format.
5. DPO only the Judge / Span Selector.
6. Validate against original qwen3.5:9b and SFT baseline.
7. Add DPO Proponent only if Judge DPO improves validation.
8. Do not train Opponent or controller until boundary metrics are stable.
9. Do not run final test until config is frozen.
```

---

## 15.18 Relation to RLVR and GRPO

DPO and RLVR are complementary.

DPO uses pairwise preference:

```text
chosen span verdict > rejected span verdict
```

RLVR uses deterministic verifiable reward:

```text
EM / F1 / boundary overlap / extractiveness
```

In this framework, RLVR supplies the reward used to build DPO pairs. After a strong DPO baseline exists, the system may use GRPO/RLVR to sample multiple trajectories and construct new preference pairs.

Recommended order:

```text
SFT → Offline Span DPO → Online DPO / GRPO with verifiable span rewards
```

Not recommended as first step:

```text
PPO over full multi-agent debate trajectories
```

PPO is more suitable later for adaptive debate controller learning, not for the first span-selection policy update.

---

# 16. RLVR for Extractive QA

RLVR is especially suitable for Phase 1 because rewards are verifiable without a complex legal proof engine.

Verifier signals:

```text
- exact match
- token F1
- character IoU
- boundary start distance
- boundary end distance
- extractiveness
- answer type match
```

Training pipeline:

```text
Generate candidate/debate trajectory
    ↓
Judge selects span
    ↓
Compare with gold answer
    ↓
Compute EM/F1/boundary reward
    ↓
Build preference pair
    ↓
DPO / GRPO / RLVR optimization
```

Preference pair example:

```json
{
  "prompt": "Question + context + candidate graph",
  "chosen": {
    "answer": "15 năm",
    "reason": "upper bound only"
  },
  "rejected": {
    "answer": "bị phạt tù từ 07 năm đến 15 năm",
    "reason": "over-extracted range"
  },
  "reward_delta": 0.42
}
```

---

# 17. Memory Graph for Phase 1

Memory remains three-tier, but content becomes span-centric.

```text
Memory buckets:
- regulations
- experiences
- cases
```

## 16.1 Regulations memory

Stores rule / article / answer policy.

```json
{
  "type": "regulation",
  "law_name": "BLHS",
  "article": "Điều 173",
  "proposition_id": "theft_penalty_upper_bound",
  "answer_policy": "If asked maximum penalty duration, extract upper bound only."
}
```

## 16.2 Experience memory

Stores reusable boundary strategies.

```json
{
  "type": "experience",
  "failure_type": "over_extraction_range",
  "lesson": "For maximum-duration questions, prefer the upper-bound span over the whole penalty range."
}
```

## 16.3 Case memory

Stores case-specific span behavior.

```json
{
  "type": "case",
  "case_id": "vilqa-331",
  "question_type": "maximum_penalty",
  "answer_type": "duration",
  "gold_span": "15 năm",
  "bad_span": "bị phạt tù từ 07 năm đến 15 năm",
  "failure_type": "over_extraction",
  "lesson": "Extract only upper bound when question asks maximum."
}
```

## 16.4 Memory governance

To prevent leakage:

```text
- Do not update memory with test gold answers.
- Snapshot memory per experiment run.
- Separate train-memory, validation-memory, and test-inference memory.
- Reflection must not copy gold answer into future agent context unless the setting explicitly allows supervised training.
- Store lessons as policies, not raw gold-answer shortcuts.
```

---

# 18. Final Output Contract

All Phase 1 methods should output the same schema.

```json
{
  "case_id": "vilqa-331",
  "question": "Mức phạt tù cao nhất là bao lâu?",
  "answer": "15 năm",
  "prediction": "15 năm",
  "selected_span_id": "S3",
  "source_doc_id": "ctx_vilqa_331",
  "start_offset": 452,
  "end_offset": 458,
  "source_sentence": "bị phạt tù từ 07 năm đến 15 năm",
  "answer_type": "duration",
  "confidence": 0.91,
  "method": "judge_mediated_span_debate",
  "verification": {
    "L0_extractiveness": "pass",
    "L1_boundary": "pass",
    "L2_relevance": "pass",
    "L3_sufficiency": "pass",
    "L4_dominance": "pass"
  },
  "failure_type": null
}
```

If offset cannot be found:

```json
{
  "answer": "15 năm",
  "source_doc_id": null,
  "start_offset": null,
  "end_offset": null,
  "verification": {
    "L0_extractiveness": "unknown"
  }
}
```

But the target behavior is to always recover source offsets for extractive answers.

---

# 19. Evaluation

## 18.1 Main metrics

```text
Exact Match
Token F1
```

## 18.2 Additional metrics

```text
character IoU
boundary start error
boundary end error
extractiveness rate
over-extraction rate
under-extraction rate
wrong-region rate
answer-type mismatch rate
fallback rate
```

## 18.3 Error taxonomy

| Error type | Meaning |
|---|---|
| `over_extraction` | predicted span is longer than gold |
| `under_extraction` | predicted span misses necessary information |
| `wrong_boundary_prefix` | unnecessary prefix included |
| `wrong_boundary_suffix` | unnecessary suffix included |
| `wrong_region` | answer extracted from wrong sentence / article |
| `paraphrase` | answer does not appear verbatim |
| `wrong_answer_type` | answer type does not match question |
| `json_parse_failure` | malformed output |
| `no_candidate_found` | candidate generator failed |
| `ignored_dominant_candidate` | judge ignored a better span |

## 18.4 Boundary-aware evaluation

For each prediction:

```text
1. Check normalized EM.
2. Compute token F1.
3. Locate predicted answer in context.
4. Compute character IoU against gold span if offsets available.
5. Classify boundary error.
6. Log source sentence and selected_span_id.
```

---

# 20. Inference-Time Compute Scaling

Framework cũ uses Debate Search Tree. For extractive QA, search should happen over candidate space instead of debate space.

```text
Candidate Search:
- Generate N candidates
- Score candidates
- Compare top-k candidates
- Run one boundary debate
- Run L4 dominance search among top-k
- Select final span
```

Recommended default:

```yaml
debate:
  max_rounds: 1
  adaptive_rounds: true
  early_stop_confidence: 0.85
  extra_round_conditions:
    - overlapping_top_candidates
    - low_confidence
    - answer_type_uncertain
    - unresolved_dominance
```

Principle:

```text
Use more candidates before using more debate rounds.
```

---

# 21. Mapping from SE-NESMAD v2 to SE-NESMAD-EQA v3

| SE-NESMAD v2 | SE-NESMAD-EQA v3 |
|---|---|
| Claim | SpanCandidate |
| Argument Graph | Span Candidate Graph |
| support / attack | contains / overlaps / refines / dominates / conflicts |
| QBAF dialectical strength | Candidate dominance score |
| Proof Verification | Span Verification |
| L0 syntax | L0 extractiveness |
| L1 consistency | L1 boundary validity |
| L2 rule satisfaction | L2 relevance / answer-type match |
| L3 proof derivation | L3 sufficiency |
| L4 counterexample search | L4 dominance search |
| Proposer solution | Proponent selected span |
| Challenger counterexample | Opponent better span / boundary challenge |
| Judge verdict | Judge span selection with offsets |
| R_claim | R_span |
| proof_validity | EM / F1 / char IoU / boundary score |
| Memory case outcome | span policy + boundary failure |
| RLVR from symbolic verifier | RLVR from gold span metrics |

---

# 22. Implementation Plan

## 21.1 New schemas

Add to `src/models.py`:

```text
AnswerTypeProfile
SpanCandidate
SpanRelation
SpanCandidateGraph
SpanVerificationResult
SpanJudgeVerdict
SpanErrorAnalysis
```

## 21.2 New modules

```text
src/span/answer_type_parser.py
src/span/candidate_generator.py
src/span/span_graph.py
src/span/span_verifier.py
src/span/dominance.py
src/span/boundary_metrics.py
```

## 21.3 Judge changes

`JudgeAgent.render_verdict()` should return:

```text
answer
selected_span_id
source_doc_id
start_offset
end_offset
source_sentence
answer_type
verification
confidence
```

## 21.4 Agent changes

Proponent / Opponent outputs must reference `span_id`.

Rules:

```text
- No free-form answer without source offset.
- New answer must be converted into SpanCandidate.
- Opponent must specify target_span_id and attack_type.
```

## 21.5 Evaluator changes

Keep EM/F1. Add:

```text
char_iou
boundary_start_error
boundary_end_error
extractiveness_rate
over_extraction_rate
under_extraction_rate
wrong_region_rate
answer_type_mismatch_rate
```

## 21.6 Memory changes

Memory reflection should write:

```text
question_type
answer_type
selected_span
failure_type
span_policy_lesson
```

not:

```text
claim verified / refuted
```

## 21.7 Ablation plan

New ablations:

```text
candidate_generator on/off
span_graph on/off
L4 dominance search on/off
answer_type_parser on/off
rounds 1 vs adaptive
memory span-policy on/off
retrieval localization on/off
PYTHEN localization on/off
```

---

# 23. Definition of Done

SE-NESMAD-EQA v3 is complete for Phase 1 when:

```text
1. Every prediction has answer + source_doc_id + offset if extractive.
2. Every case has SpanCandidateGraph artifact.
3. Judge selects from candidate graph, not free-form text.
4. L0-L4 Span Verification runs for selected span.
5. Metrics include EM/F1 plus extractiveness and boundary errors.
6. Error analysis classifies over/under-extraction and wrong boundary.
7. Adaptive rounds do not underperform fixed round 1 on validation.
8. Memory update does not leak gold answer.
9. Test split is evaluated only after validation config is frozen.
10. All baselines share the same output contract.
```

---

# 24. Recommended Phase 1 Pipeline

Final recommended pipeline:

```text
Input: Legal Context + Question
  ↓
Answer-Type Parser
  ↓
Retrieval / Localization
  - BM25
  - optional rerank
  - optional PYTHEN localization
  ↓
Span Candidate Generator
  ↓
Span Candidate Graph
  ↓
One-Round Boundary Debate
  - Proponent selects span
  - Opponent challenges boundary
  - Judge selects final span
  ↓
Span Verification Ladder
  - L0 extractiveness
  - L1 boundary
  - L2 relevance
  - L3 sufficiency
  - L4 dominance
  ↓
Final Answer
  - text
  - source_doc_id
  - start_offset
  - end_offset
  ↓
Evaluation
  - EM
  - F1
  - boundary metrics
  - optional LLM-as-judge
  ↓
Reflection / Memory
  - span policy
  - failure type
  - reusable extraction lesson
```

---

# 25. Short Report Version

SE-NESMAD-EQA is a span-centric neuro-symbolic multi-agent framework for Vietnamese legal extractive question answering. Unlike the original claim-centric SE-NESMAD framework, which models legal reasoning as claim verification over an argument graph, SE-NESMAD-EQA treats the answer span as the central object. The system first retrieves and localizes relevant legal text, optionally using symbolic rule localization to identify applicable provisions. It then generates span candidates from retrieved contexts using regex, answer-type parsing, sentence windows, and LLM proposals. These candidates are organized into a Span Candidate Graph whose edges represent textual and boundary relations such as containment, overlap, refinement, conflict, and dominance. A constrained multi-agent debate is then performed: the Proponent proposes the best span, the Opponent challenges boundary and sufficiency errors, and the Judge selects the final extractive answer with source offsets. Verification is reformulated from proof derivation to span validation, consisting of extractiveness, boundary validity, relevance, sufficiency, and dominance search. The framework supports verifiable rewards through exact match, token F1, character overlap, and boundary accuracy, enabling RLVR-style optimization without requiring a full logical proof engine for Phase 1.

---

# 26. Key Takeaway

The original SE-NESMAD framework remains valuable for legal reasoning and courtroom judgment tasks. But for Phase 1 extractive QA, the framework must be re-centered around answer spans.

```text
Claim-centric reasoning
    ↓
Span-centric extraction
```

The system should not primarily ask:

```text
Which claim is legally true?
```

It should ask:

```text
Which text span is the best extractive answer?
```

This shift preserves the strengths of the old framework — multi-agent reasoning, memory, evaluator independence, and RLVR — while making the architecture fit the actual evaluation target of ViLQA / ALQAC: exact, minimal, grounded legal answer spans.
