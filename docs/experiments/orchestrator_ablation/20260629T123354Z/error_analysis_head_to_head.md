# Orchestrator Ablation — Head-to-Head Error Analysis

Run: `20260629T123354Z` | Split: validation 53

## Overall

| Orchestrator | EM | Hits |
|---|---:|---:|
| fixed | 0.6038 | 32/53 |
| judge_mediated | 0.6792 | 36/53 |
| **Δ EM** | **+0.0755** | **+4 cases** |

## Head-to-head

- Both correct: 30
- Both wrong: 15
- **Judge-mediated fixes fixed's errors**: 6
- **Judge-mediated regresses fixed's hits**: 2

### Judge-mediated wins (fixed wrong → mediated correct)

- **vilqa-181**
  - gold: `ngày tiếp theo liền kề của ngày xảy ra sự kiện đó`
  - fixed: `khi thời hạn bắt đầu bằng một sự kiện thì ngày xảy ra sự kiện không được tính mà tính từ ngày tiếp theo liền kề của ngày`
  - mediated: `ngày tiếp theo liền kề của ngày xảy ra sự kiện đó`
- **vilqa-189**
  - gold: `Chiếm hữu`
  - fixed: `quyền đối với tài sản`
  - mediated: `Chiếm hữu`
- **vilqa-311**
  - gold: `phạt tù từ 01 năm đến 05 năm`
  - fixed: `b) Là chỉ huy hoặc sỹ quan; ... thì bị phạt tù từ 01 năm đến 05 năm`
  - mediated: `phạt tù từ 01 năm đến 05 năm`
- **vilqa-359**
  - gold: `07 năm`
  - fixed: `phạt tù từ 02 năm đến 07 năm`
  - mediated: `07 năm`
- **vilqa-424**
  - gold: `Ban Cơ yếu Chính phủ`
  - fixed: `Ban Cơ yếu Chính phủ thẩm định an ninh mạng đối với hệ thống thông tin cơ yếu thuộc Ban Cơ yếu Chính phủ`
  - mediated: `Ban Cơ yếu Chính phủ`
- **vilqa-453**
  - gold: `gây thiệt hại cho người khác, để vi phạm nghĩa vụ của mình hoặc thực hiện mục đích khác trái pháp luật.`
  - fixed: `vi phạm nghĩa vụ của mình hoặc thực hiện mục đích khác trái pháp luật`
  - mediated: `gây thiệt hại cho người khác, để vi phạm nghĩa vụ của mình hoặc thực hiện mục đích khác trái pháp luật`

### Judge-mediated regressions (fixed correct → mediated wrong)

- **vilqa-36**
  - gold: `03 năm`
  - fixed: `03 năm`
  - mediated: `đến 03 năm`
- **vilqa-40**
  - gold: `từ 03 năm đến 07 năm`
  - fixed: `từ 03 năm đến 07 năm`
  - mediated: `phạt tù từ 03 năm đến 07 năm`

## Error taxonomy (errors only)

| Type | fixed | judge_mediated |
|---|---:|---:|
| OTHER | 1 | 2 |
| OVER_EXTRACTION | 13 | 7 |
| PARTIAL_SPAN | 7 | 8 |
