# Research Context

## Vấn Đề Nghiên Cứu
**Legal Judgment Prediction (LJP - Dự đoán phán quyết pháp lý)** và **Legal Reasoning (Lập luận pháp lý)** là những tác vụ cực kỳ thách thức trong lĩnh vực AI & NLP. Lĩnh vực pháp lý đòi hỏi khả năng suy luận logic chặt chẽ, hiểu biết sâu sắc về ngữ cảnh xã hội, và khả năng áp dụng các điều luật trừu tượng vào các tình huống thực tế phức tạp.
Các phương pháp đơn tác tử (Single-Agent) hoặc mô hình phân loại truyền thống thường gặp phải các vấn đề:
- **Hallucination (Ảo tưởng)**: Áp dụng sai điều luật hoặc tự bịa đặt ra các tình tiết không có trong hồ sơ vụ án.
- **Định kiến một chiều**: Chỉ tập trung vào các chứng cứ buộc tội mà bỏ qua các yếu tố bào chữa/giảm nhẹ (hoặc ngược lại).
- **Thiếu tính giải thích**: Đưa ra phán quyết trực tiếp mà không giải thích rõ ràng quá trình lập luận biện chứng.

## Related Work
1. **Single-Agent LJP & Legal NLP**: 
   - Sử dụng các mô hình ngôn ngữ lớn (LLM) thông qua phương pháp Few-shot Prompting hoặc Chain-of-Thought (CoT) để dự đoán tội danh, điều luật áp dụng và mức án trực tiếp từ mô tả vụ việc.
2. **Multi-Agent Debate (Tranh biện đa tác tử)**:
   - Các nghiên cứu gần đây (ví dụ: Du et al., 2023; Liang et al., 2023) chứng minh rằng việc cho phép các tác tử AI thảo luận, phản biện và hợp tác với nhau sẽ giúp cải thiện đáng kể khả năng lập luận, sửa lỗi logic và tăng chất lượng đầu ra cho các tác vụ phức tạp.
3. **Simulated Courtrooms (Giả lập phiên tòa)**:
   - Các nghiên cứu sơ khởi ứng dụng đa tác tử vào pháp luật (ví dụ: AgenticSimLaw) khám phá cách thức mô phỏng hành vi của luật sư và kiểm sát viên trong một môi trường giả định để phân tích luật pháp.

## Research Gap
Hầu hết các nghiên cứu LJP hiện tại coi tác vụ này như là một bài toán phân loại văn bản một chiều hoặc suy luận tuyến tính. Trong thực tế, hệ thống tư pháp hoạt động dựa trên **nguyên tắc tranh tụng dân chủ và biện chứng (Adversarial Debate)** giữa hai bên có lợi ích đối nghịch (Bên buộc tội và Bên gỡ tội) dưới sự phân xử của một bên thứ ba độc lập (Thẩm phán). 
Hiện chưa có nhiều nghiên cứu tập trung vào việc thiết kế một **khung giả lập tranh tụng đa tác tử có cấu trúc** để giải quyết bài toán suy luận pháp lý này, nhằm đánh giá xem liệu quá trình tranh biện hai chiều có thực sự giúp đưa ra phán quyết khách quan và chính xác hơn hay không.

## Phương Pháp Đề Xuất (Adversarial Courtroom Simulation)
Dự án đề xuất một framework đa tác tử mô phỏng phiên tòa gồm các thành phần:
- **Prosecutor Agent (Kiểm sát viên)**: Có xu hướng buộc tội, nhấn mạnh các chứng cứ buộc tội, tình tiết tăng nặng và đề xuất mức án cao nhất phù hợp với luật.
- **Defense Agent (Luật sư bào chữa)**: Có xu hướng gỡ tội hoặc giảm nhẹ, nhấn mạnh các tình tiết giảm nhẹ (thành khẩn khai báo, hoàn cảnh, bồi thường), chỉ ra kẽ hở hoặc điểm nghi vấn trong chứng cứ buộc tội.
- **Defendant Agent (Bị cáo)**: Cung cấp lời khai cá nhân, trạng thái tâm lý, hoàn cảnh hành vi dưới dạng hội thoại góc nhìn thứ nhất.
- **Judge Agent (Thẩm phán)**: Đóng vai trò điều phối, duy trì quy trình tố tụng, lắng nghe lập luận từ cả hai bên, yêu cầu làm rõ khi cần, phân tích các điều luật một cách khách quan nhất và đưa ra bản án cuối cùng.

## Dataset
- **Phase 1 hiện tại**: ViLQA/ALQAC legal question answering từ `data/ALQAC.csv`. Mỗi mẫu gồm `context`, `question`, `answer`; task là dự đoán câu trả lời ngắn/trích xuất và đánh giá bằng Exact Match/F1.
- **Phase 2+ dự kiến**: Hồ sơ vụ án hình sự/dân sự có cấu trúc gồm fact, evidence, testimonies, applicable law và judgment ground truth cho Legal Judgment Prediction.
- **Ràng buộc dữ liệu**: Split train/validation/test phải cố định; retrieval/memory không được fit/tune trên validation hoặc test labels.

## Baselines So Sánh
1. **Direct Prediction (Single-Agent)**: Đưa toàn bộ hồ sơ vụ án vào một LLM đơn lẻ và yêu cầu nó đưa ra tội danh, điều luật áp dụng và mức án trực tiếp.
2. **Chain of Thought (CoT - Single-Agent)**: Yêu cầu một LLM lập luận từng bước (phân tích cáo trạng -> xem xét bằng chứng -> áp dụng điều luật -> ra phán quyết) trước khi đưa ra kết luận.
3. **Vanilla Debate**: Hai tác tử tranh luận tự do không phân vai cụ thể và không có quy trình tố tụng kiểm soát, sau đó đưa ra kết luận đồng thuận.
4. **Structured Debate (Phase 1, config chính)**: `proponent` và `opponent` tranh luận; **JudgeAgent điều phối** (`judge_mediated`) chọn lượt tiếp theo, cập nhật belief, và đưa verdict JSON. `fixed` orchestrator (turn order Python) chỉ dùng cho ablation legacy.
5. **Extractive QA Reader**: mBERT/PhoBERT/XLM-R style reader trên context để tạo strong floor cho QA.
6. **BM25 + Reader**: Retrieve top-k context bằng BM25-lite rồi đưa vào reader, tương ứng hướng RAG trong AgentsCourt.

## Phân Kỳ Nghiên Cứu
- **Phase 1 = ViLQA QA Debate**: Mục tiêu là kiểm chứng liệu debate/LLM thật có cải thiện câu trả lời legal QA so với Direct/CoT/reader baseline. Đây là bước bắt buộc trước khi claim về courtroom reasoning.
- **Phase 2+ = Courtroom LJP**: Sau khi Phase 1 có kết quả hợp lệ, mở rộng sang multi-agent courtroom simulation với Prosecutor/Defense/Defendant/Judge, protocol tố tụng và schema phán quyết pháp lý.

## Nguyên Tắc Đánh Giá
- Không tune prompt, model, temperature hoặc reader hyperparameter trên test split.
- Báo cáo model/provider/temperature theo method trong `metrics.json`.
- Theo dõi `fallback_rate` của judge; fallback cao làm giảm giá trị kết luận về debate.
- Không claim cải thiện nếu chưa có metric trên cùng split và cùng evaluation code.

## Thiết Kế P1 Theo Related Work
- **AgentsCourt/RAG**: retrieval pipeline gồm BM25 rough top-n, optional semantic rerank, và external legal corpus UTS_VLC để evidence không chỉ đến từ ALQAC train contexts.
- **MASER / AgentCourt-AdvEvol**: memory tách `regulations`, `experiences`, `cases`; hỗ trợ read-only hoặc read+update, reflection prompt, dedup/limit, và embedding retrieval.
- **LLM-as-judge evaluation**: EM/F1 là automated metrics có gold answer; legal accuracy, argument quality, logical consistency là rubric evaluation không đưa gold answer vào prompt.
- **AgenticSimLaw / Courtroom-LLM / AgentsCourt**: debate loop có **judge điều phối** (`judge_mediated` — config chính), closing statements, optional judge follow-up, early stopping theo confidence.
- **Ablation study**: biến chính gồm retrieval (`off/bm25_only/bm25_rerank`), memory (`off/read_only/read_update`), rounds (`1/3/5`), judge (`off/on`), roles (`proponent-opponent/prosecutor-defense`). Phase 1 chỉ implement role `proponent-opponent`; `prosecutor-defense` dành cho Phase 2 courtroom/LJP.
