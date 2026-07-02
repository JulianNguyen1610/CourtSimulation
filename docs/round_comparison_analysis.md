# Phân Tích Sự Khác Biệt Giữa Các Số Round (1, 3, 5)

## Tổng Quan

**Dataset:** ALQAC validation split (53 cases)  
**Model:** qwen3.5:9b (local Ollama)  
**Cấu hình:** retrieval=bm25_only, memory=read_only, judge=on, orchestrator=judge_mediated  
**Ngày thực nghiệm:** 2026-06-19

## Kết Quả Chính

| Metric | Round 1 | Round 3 | Round 5 |
|--------|---------|---------|---------|
| **Exact Match** | **0.5660** (30/53) | 0.4717 (25/53) | 0.5283 (28/53) |
| **F1 Score** | **0.8564** | 0.7983 | 0.8373 |
| **Parse Attempts** | 106 | 212 | 318 |
| **Fallback Rate** | 1.89% | 2.36% | 0.94% |

### Xu Hướng Hiệu Suất

#### Exact Match
- **Round 1 → Round 3:** Giảm **9.43 pp** (từ 56.60% xuống 47.17%) ❌
- **Round 3 → Round 5:** Tăng **5.66 pp** (từ 47.17% lên 52.83%) ✓
- **Round 1 → Round 5:** Giảm ròng **3.77 pp** (từ 56.60% xuống 52.83%) ❌

#### F1 Score  
- **Round 1 → Round 3:** Giảm **5.81 pp** (từ 0.8564 xuống 0.7983) ❌
- **Round 3 → Round 5:** Tăng **3.90 pp** (từ 0.7983 lên 0.8373) ✓
- **Round 1 → Round 5:** Giảm ròng **1.91 pp** (từ 0.8564 xuống 0.8373) ❌

#### Fallback Rate
- Round 1 → Round 3: Tăng 0.47 pp (model kém ổn định hơn)
- Round 3 → Round 5: Giảm 1.42 pp (model ổn định lại)
- Round 5 có fallback rate thấp nhất (0.94%)

## Phân Tích Chi Tiết

### 1. Round 1 Vượt Trội (Không Như Mong Đợi)

**Kết luận chính:** Debate đơn giản (1 round) cho kết quả tốt nhất, cả về EM và F1.

**Giả thuyết:**
- Judge có thể tổng hợp thông tin từ 1 round tranh luận mà không bị nhiễu
- Quyết định ban đầu dựa trên context gốc + 1 lượt tranh luận là tối ưu
- Độ phức tạp tăng không tỷ lệ thuận với chất lượng kết quả

### 2. Round 3 Kém Nhất (Điểm Yếu)

**Kết luận chính:** 3 rounds là điểm thấp nhất cả về EM và F1.

**Giả thuyết:**
- Tranh luận kéo dài làm tăng noise trong thông tin
- Judge bị phân tâm bởi nhiều luận điểm mâu thuẫn
- Agents có thể bắt đầu "hallucinate" hoặc tạo ra luận điểm yếu hơn ở các round sau
- Context window bị lấp đầy bởi các turn trước, ảnh hưởng đến reasoning

### 3. Round 5 Phục Hồi Một Phần

**Kết luận chính:** 5 rounds tốt hơn 3 rounds nhưng vẫn kém round 1.

**Giả thuyết:**
- Với đủ nhiều rounds, agents có thể hội tụ về câu trả lời tốt hơn
- Fallback rate thấp nhất (0.94%) → model ổn định với nhiều vòng
- Tuy nhiên, cost của nhiều rounds vẫn không bù được benefit

## So Sánh Case Cụ Thể

### Cases Round 1 Đúng, Round 3 Sai (5 cases)

#### Case vilqa-352: Lỗi Over-Extraction
- **Gold:** "100.000.000 đồng"
- **R1 pred:** "100.000.000 đồng" ✓
- **R3 pred:** "phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng" ❌
- **Phân tích:** R3 thêm prefix không cần thiết, làm mất exact match

#### Case vilqa-236: Lỗi Prefix
- **Gold:** "01 tháng"
- **R1 pred:** "01 tháng" ✓
- **R3 pred:** "Sau 01 tháng" ❌
- **Phân tích:** R3 thêm từ "Sau", gây mất EM

#### Case vilqa-125: Lỗi Expansion
- **Gold:** "do hai bên thỏa thuận"
- **R1 pred:** "do hai bên thỏa thuận" ✓
- **R3 pred:** "theo điều kiện do hai bên thỏa thuận" ❌
- **Phân tích:** R3 mở rộng câu trả lời không cần thiết

#### Case vilqa-331: Lỗi Over-Extraction
- **Gold:** "15 năm"
- **R1 pred:** "15 năm" ✓
- **R3 pred:** "bị phạt tù từ 07 năm đến 15 năm" ❌
- **Phân tích:** R3 bao gồm cả range thay vì chỉ upper bound

#### Case vilqa-189: Lỗi Definition Expansion
- **Gold:** "Chiếm hữu"
- **R1 pred:** "Chiếm hữu" ✓
- **R3 pred:** "Chiếm hữu là việc chủ thể nắm giữ..." ❌
- **Phân tích:** R3 trả về định nghĩa thay vì chỉ thuật ngữ

**Pattern chung:** Round 3 có xu hướng thêm context/prefix hoặc mở rộng câu trả lời, dẫn đến over-extraction.

### Cases Round 3 Đúng, Round 1 Sai (ít nhất 2 cases)

#### Case vilqa-36: Under-Extraction
- **Gold:** "03 năm"
- **R1 pred:** "phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm" ❌
- **R3 pred:** "03 năm" ✓
- **Phân tích:** R1 over-extract, R3 đúng extractive

#### Case vilqa-443: Long Answer
- **Gold:** "quốc phòng, an ninh quốc gia, trật tự, an toàn xã hội, đạo đức xã hội, sức khỏe của cộng đồng."
- **R1 pred:** (toàn bộ câu dài từ context) ❌
- **R3 pred:** (chính xác gold) ✓
- **Phân tích:** R1 extract quá nhiều cho câu hỏi dài

**Pattern chung:** Round 1 đôi khi over-extract cho câu hỏi phức tạp, Round 3 có khả năng refine tốt hơn trong một số trường hợp này.

## Trade-offs Giữa Các Round

### Round 1
**Ưu điểm:**
- Hiệu suất cao nhất (EM 0.5660, F1 0.8564)
- Đơn giản, nhanh (106 parse attempts)
- Ít bị nhiễu bởi tranh luận phức tạp

**Nhược điểm:**
- Một số case phức tạp bị over-extract
- Không có cơ hội refine qua nhiều vòng

### Round 3
**Ưu điểm:**
- Có thể refine tốt hơn một số case phức tạp
- Debate đủ dài để explore các góc nhìn

**Nhược điểm:**
- Hiệu suất thấp nhất (EM 0.4717, F1 0.7983)
- Xu hướng over-extract/thêm prefix
- Fallback rate cao hơn round 1
- Cost tính toán gấp đôi round 1

### Round 5
**Ưu điểm:**
- Ổn định nhất (fallback rate 0.94%)
- Phục hồi một phần so với round 3
- Hội tụ tốt hơn với nhiều rounds

**Nhược điểm:**
- Vẫn kém round 1 (EM -3.77 pp, F1 -1.91 pp)
- Cost tính toán gấp 3 lần round 1
- Không justify được cost/benefit

## Kết Luận và Khuyến Nghị

### Kết Luận Chính

1. **Hiệu ứng phi tuyến:** Tăng số round KHÔNG cải thiện kết quả theo cách đơn điệu
2. **Round 1 là tối ưu:** Đơn giản nhưng hiệu quả nhất cho ALQAC validation
3. **Round 3 là điểm yếu:** Cả EM và F1 đều thấp nhất, không nên sử dụng
4. **Round 5 không justify cost:** Tốt hơn round 3 nhưng vẫn kém round 1

### Giải Thích Hiện Tượng

**Tại sao Round 1 tốt nhất?**
- Context ban đầu + 1 lượt tranh luận đủ thông tin cho judge
- Quyết định đơn giản ít bị nhiễu hơn
- Debate ngắn giữ được focus vào câu hỏi chính

**Tại sao Round 3 kém?**
- Tranh luận kéo dài làm tăng noise
- Agents có thể đi xa khỏi câu hỏi gốc
- Judge phải xử lý nhiều thông tin mâu thuẫn
- Over-extraction do cố gắng tổng hợp nhiều luận điểm

**Tại sao Round 5 phục hồi?**
- Agents có thể hội tụ sau nhiều rounds
- Model ổn định hơn (fallback rate thấp)
- Nhưng vẫn không bù được noise tích lũy

### Khuyến Nghị

1. **Sử dụng Round 1 làm baseline chính** cho các thí nghiệm tiếp theo
2. **Không nên dùng Round 3** - worst case trong cả 3 cấu hình
3. **Round 5 chỉ dùng khi cần stability** (low fallback rate) nhưng chấp nhận trade-off về accuracy
4. **Cần kiểm tra trên test split** để xác nhận xu hướng này generalize

### Hướng Nghiên Cứu Tiếp Theo

1. **Adaptive rounds:** Stop debate khi judge đủ confident (early stopping)
2. **Prompt tuning:** Hướng dẫn agents tránh over-extraction ở round 3+
3. **Weighted rounds:** Judge ưu tiên thông tin từ round đầu, giảm trọng số các round sau
4. **Test split validation:** Kiểm tra pattern này có đúng trên test không

## Appendix: Thống Kê Bổ Sung

### Performance Breakdown

| Metric | R1 vs R3 | R3 vs R5 | R1 vs R5 |
|--------|----------|----------|----------|
| EM delta (pp) | -9.43 | +5.66 | -3.77 |
| F1 delta (pp) | -5.81 | +3.90 | -1.91 |
| Cases flipped R1→R3 | 5+ | - | - |
| Cases flipped R3→R1 | 2+ | - | - |
| Net EM change | -5 cases | +3 cases | -2 cases |

### Cost Analysis

| Round | Parse Attempts | Relative Cost | EM | Cost-Efficiency |
|-------|----------------|---------------|----|-----------------| 
| 1 | 106 | 1.0x | 0.5660 | Baseline (best) |
| 3 | 212 | 2.0x | 0.4717 | 0.47x (worst) |
| 5 | 318 | 3.0x | 0.5283 | 0.35x (poor) |

**Cost-Efficiency = (EM / Relative Cost) normalized to Round 1 = 1.0**

Round 1 có cost-efficiency cao nhất, Round 5 kém nhất mặc dù EM khá hơn Round 3.

---

**Tác giả:** Auto-generated analysis  
**Ngày:** 2026-07-01  
**Nguồn dữ liệu:** outputs/p1_ablation_matrix/20260619T034329Z/
