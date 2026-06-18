# Project Brief

## Chủ Đề Nghiên Cứu
**Multi-Agent Courtroom Simulation Framework (Khung giả lập phiên tòa đa tác tử)**. Dự án tập trung vào việc nghiên cứu và xây dựng một hệ thống gồm nhiều tác tử AI đóng các vai trò khác nhau trong một phiên tòa (Thẩm phán, Kiểm sát viên/Viện kiểm sát, Luật sư bào chữa, Bị cáo/Bị hại, Nhân chứng) để tự động hóa việc tranh luận, phân tích tình huống pháp lý và đưa ra phán quyết.

## Câu Hỏi Nghiên Cứu
1. **Khả năng mô phỏng**: Làm thế nào để thiết kế các vai trò tác tử AI có hành vi, động cơ và ngôn ngữ lập luận pháp lý trung thực với thực tế pháp lý (ví dụ: Luật sư bào chữa tìm giảm nhẹ hình phạt, Kiểm sát viên buộc tội đúng người đúng tội)?
2. **Hiệu quả lập luận**: Việc giả lập tranh luận đa tác tử (Multi-Agent Debate) có giúp nâng cao độ chính xác, tính khách quan và giảm thiểu định kiến (bias) trong việc dự đoán phán quyết (Legal Judgment Prediction) so với phương pháp đơn tác tử (Single-Agent)?
3. **Quy trình tố tụng**: Làm thế nào để kiểm soát luồng tương tác và giao thức tranh luận giữa các tác tử tuân thủ chặt chẽ theo các bước của một phiên tòa thực tế (Cáo trạng -> Tranh tụng -> Nghị án -> Tuyên án)?

## Mục Tiêu
- **Mục tiêu chính**: Xây dựng một framework đa tác tử mô phỏng phiên tòa linh hoạt, có khả năng chạy trên các tập hồ sơ vụ án để tạo ra các cuộc tranh luận pháp lý tự động và đưa ra phán quyết có căn cứ.
- **Mục tiêu phụ**:
  - Thiết kế và tối ưu hóa hệ thống Prompt cho từng vai trò pháp lý.
  - Xây dựng cơ chế đánh giá (Evaluation Metric) chất lượng tranh luận pháp lý và độ chính xác của phán quyết.
  - So sánh hiệu năng của phương pháp đa tác tử tranh luận với baseline đơn tác tử (ví dụ: GPT hoặc LLM phân tích trực tiếp hồ sơ vụ án).

## Phạm Vi
- **Trong phạm vi**:
  - Phát triển mã nguồn lõi cho framework đa tác tử (Agent, Courtroom/Session Controller, Debate Protocol).
  - Định nghĩa 4 vai trò cốt lõi: **Thẩm phán (Judge)**, **Kiểm sát viên/Bên buộc tội (Prosecutor)**, **Luật sư bào chữa (Defense Counsel)**, và **Bị cáo (Defendant)**.
  - Sử dụng các hồ sơ vụ án mẫu (dataset) về hình sự hoặc dân sự để chạy thử nghiệm.
  - Đánh giá chất lượng lập luận và độ chính xác của phán quyết dựa trên luật pháp hiện hành (ví dụ: Bộ luật Hình sự).
- **Ngoài phạm vi**:
  - Không xây dựng giao diện người dùng (UI) 3D hoặc audio/video giả lập; giao diện đầu ra là text, terminal log, và báo cáo phân tích (Markdown).
  - Không tích hợp với hệ thống tư pháp thực tế; hệ thống chỉ mang tính chất nghiên cứu học thuật và hỗ trợ phân tích.

## Tiêu Chí Thành Công
- Hệ thống chạy không bị lỗi vòng lặp hội thoại hoặc mất kiểm soát ngữ cảnh (hallucination).
- Thẩm phán đưa ra được bản án cuối cùng trích dẫn đúng điều khoản luật và có lập luận logic dựa trên diễn biến tranh luận.
- Kết quả phán quyết (tội danh, khung hình phạt) có sự cải thiện về tính hợp lý và độ chính xác so với việc để một LLM đơn lẻ phân tích trực tiếp hồ sơ vụ án.

## Ràng Buộc
- **Tài nguyên**: Giới hạn về API key và chi phí gọi mô hình ngôn ngữ lớn (OpenAI, Anthropic, Gemini hoặc các mô hình mã nguồn mở như Llama/Qwen).
- **Ngữ cảnh**: Chiều dài ngữ cảnh (Context Window) bị giới hạn khi cuộc tranh luận diễn ra qua nhiều lượt đối thoại dài.
- **Dữ liệu**: Hồ sơ vụ án bằng tiếng Việt hoặc tiếng Anh cần được ẩn danh hóa và chuẩn hóa định dạng đầu vào.
