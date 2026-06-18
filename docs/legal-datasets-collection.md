# Legal / Law Public Datasets & Benchmarks

Tổng hợp các dataset công khai dạng QA, Fact-checking, Benchmark, và Corpus trong lĩnh vực pháp lý (Legal NLP).

---

## 1. Legal QA (Hỏi-Đáp Pháp lý)

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **LegalBench** | 162 task đo legal reasoning (rule application, issue-spotting, statutory interpretation...) | 20,000+ examples | [HuggingFace](https://huggingface.co/datasets/nguha/legalbench) \| [Paper (NeurIPS 2023)](https://openreview.net/forum?id=WqSPQFxFRC) |
| **LegalBench-RAG** | Benchmark đánh giá hệ thống RAG trên hợp đồng pháp lý | Focused | [GitHub](https://github.com/zeroentropy-ai/legalbenchrag) |
| **LEXam** | Benchmark trên 340 đề thi luật thực tế (ICLR 2026) | 340 exams | [Project Page](https://lexam-benchmark.github.io/) |
| **VLQA** (Vietnamese) | 3,000+ câu hỏi pháp lý thực tế VN, gắn với ~59,000 điều luật | 3,000+ QA pairs | [Paper](https://arxiv.org/abs/2507.19995) |
| **ObliQA** | QA về nghĩa vụ pháp lý từ quy định tài chính ADGM | Specialized | [GitHub](https://github.com/Regulatory-NLP/ObliQA) |

---

## 2. Legal Fact-Checking & Verification

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **CaseFacts** (2026) | Verify claims về án lệ US Supreme Court: Supported / Refuted / Overruled | 6,294 claims | [GitHub](https://github.com/idirlab/CaseFacts) \| [HuggingFace](https://huggingface.co/datasets/IDIRLab/CaseFacts) \| [Paper](https://arxiv.org/abs/2601.17230) |
| **LegalCiteBench** (2026) | Đánh giá độ tin cậy trích dẫn pháp lý của LLM (citation retrieval, error detection, case verification) | ~24,000 instances | [Paper](https://arxiv.org/abs/2605.10186) |
| **LegalHalBench** (2025) | Phát hiện 5 loại hallucination khi LLM trả lời câu hỏi pháp lý | Benchmark | [GitHub](https://github.com/YinghaoHu/LegalHalBench) \| [Paper](https://arxiv.org/abs/2501.06521) |
| **ContractNLI** | NLI cho hợp đồng (entailment / contradiction / neutral) | - | [GitHub](https://github.com/nyu-mll/contract-nli) |

---

## 3. Legal Judgment Prediction (LJP)

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **CAIL2018** | Vụ án hình sự TQ: fact → charge + article + sentence | 2.68M cases | [GitHub](https://github.com/thunlp/CAIL) \| [Download](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) \| [Paper](https://arxiv.org/abs/1807.02478) |
| **MultiJustice / MPMCP** (2025) | Đa bị cáo + đa tội danh, 4 kịch bản phức tạp | 20,000 cases | [GitHub](https://github.com/lololo-xiao/MultiJustice-MPMCP) \| [Paper](https://arxiv.org/abs/2507.06909) |
| **MUD** (ACL 2024) | Multi-defendant charge prediction với criminal element annotations | 2,865 cases | [GitHub](https://github.com/xuqi220/MUD) |
| **SimuCourt** (EMNLP 2024) | Benchmark cho courtroom simulation (hình sự + dân sự + hành chính) | 420 judgments | [Paper](https://arxiv.org/abs/2403.02959) |
| **AnnoCaseLaw** (2025) | US Appeals Court, expert-annotated, explainable LJP | 471 cases | [GitHub](https://github.com/anonymouspolar1/annocaselaw) |
| **LegalReasoner** (ACL 2025) | Step-wise verification-correction for legal judgment reasoning | 393,945 cases | [Paper](https://aclanthology.org/2025.acl-long.361.pdf) |

---

## 4. Courtroom Simulation & Multi-Agent Legal

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **SimuCourt / AgentsCourt** (EMNLP 2024) | Multi-agent courtroom debate simulation + judicial decision-making | 420 cases | [Paper v3](https://arxiv.org/abs/2403.02959) \| [PDF](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-emnlp.549.pdf) |
| **MASER / SynthLaw-4.5k** (NAACL 2025) | Multi-agent legal simulation driver + synthetic legal scenario dataset | 4,532 samples | [GitHub](https://github.com/FudanDISC/MASER) \| [HuggingFace](https://huggingface.co/datasets/ShengbinYue/SynthLaw) \| [Paper](https://aclanthology.org/2025.findings-naacl.365.pdf) |

---

## 5. Legal NLU / Multi-task Benchmark

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **LexGLUE** | Multi-task benchmark: ECtHR, EUR-LEX, LEDGAR, UNFAIR-ToS, CaseHOLD... | 7 tasks | [GitHub](https://github.com/coastalcph/lex_glue) |
| **LEXTREME** | Multilingual legal benchmark, 24 ngôn ngữ | Multi-task | [Paper](https://arxiv.org/abs/2301.13126) |
| **VLegal-Bench** (Vietnamese) | 22 task pháp lý VN: Court Decision Prediction, Penalty Estimation, Judicial Reasoning... | 22 tasks | [HuggingFace](https://huggingface.co/datasets/CMC-OPENAI/VLegal-Bench) |
| **CaseHOLD** | Multiple choice QA: chọn đúng case holding từ bối cảnh pháp lý | ~53,000 | [GitHub](https://github.com/reglab/casehold) |
| **CUAD** | Expert-annotated clause extraction từ hợp đồng | 510 contracts | [Website](https://www.atticusprojectai.org/cuad) \| [GitHub](https://github.com/TheAtticusProject/cuad) |
| **MAUD** | Expert-annotated reading comprehension cho merger agreements | 39,000+ examples | [Paper](https://arxiv.org/abs/2301.00876) |

---

## 6. Vietnamese Legal Datasets

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **congbobanan-toaan-gov-vn** | Mirror bản án từ cổng Tòa án nhân dân tối cao VN (PDF + markdown + metadata) | Full portal | [HuggingFace](https://huggingface.co/datasets/tmquan/congbobanan-toaan-gov-vn) |
| **anle-toaan-gov-vn** | Án lệ VN, cấu trúc phân cấp document → section → paragraph → sentence | Full portal | [HuggingFace](https://huggingface.co/datasets/tmquan/anle-toaan-gov-vn) |
| **UTS_VLC** | Bộ luật VN: Hình sự, Dân sự, Tố tụng... (1945–2025) | Full VN codes | [HuggingFace](https://huggingface.co/datasets/undertheseanlp/UTS_VLC) |
| **VLegal-Bench** | 22 task pháp lý VN (đã liệt kê ở mục 5) | 22 tasks | [HuggingFace](https://huggingface.co/datasets/CMC-OPENAI/VLegal-Bench) |
| **VLQA** | 3,000+ câu hỏi pháp lý thực tế VN (đã liệt kê ở mục 1) | 3,000+ QA pairs | [Paper](https://arxiv.org/abs/2507.19995) |

---

## 7. Legal Corpus (Pre-training / RAG)

| Dataset | Mô tả | Quy mô | Link |
|---|---|---|---|
| **Pile of Law** | Tổng hợp text pháp lý EN (opinions, regulations, contracts) | 256 GB | [GitHub](https://github.com/Breakend/PileOfLaw) |
| **MultiLegalPile** | Corpus pháp lý đa ngôn ngữ, 24 languages | Massive | [HuggingFace](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile) |

---

## 8. Awesome Lists (danh sách tổng hợp, cập nhật liên tục)

- [neelguha/legal-ml-datasets](https://github.com/neelguha/legal-ml-datasets) — Danh sách dataset/benchmark ML x Law
- [openlegaldata/awesome-legal-data](https://github.com/openlegaldata/awesome-legal-data) — Dataset pháp lý theo quốc gia/khu vực
- [chen-friedman/awesome-legaltech](https://github.com/chen-friedman/awesome-legaltech) — Tổng hợp dataset + model + tool

---

*Tổng hợp ngày 15/06/2026*
