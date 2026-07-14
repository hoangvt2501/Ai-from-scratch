# Phát triển Agent định hướng đánh giá

> Hướng dẫn của Anthropic: "Bắt đầu với prompts đơn giản, tối ưu hóa chúng với đánh giá toàn diện và chỉ thêm hệ thống agentic nhiều bước khi cần thiết." Đánh giá không phải là bước cuối cùng. Đó là vòng lặp bên ngoài thúc đẩy mọi lựa chọn khác trong Giai đoạn 14.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Tất cả Giai đoạn 14.
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho ba lớp đánh giá - benchmarks tĩnh, ngoại tuyến tùy chỉnh, production trực tuyến - và mỗi lớp dùng để làm gì.
- Giải thích vòng lặp chặt chẽ optimizer người đánh giá.
- Mô tả phương pháp hay nhất năm 2026: đánh giá trực tiếp bên cạnh mã, chạy trong CI, cổng PR.
- Kết nối mọi bài học Giai đoạn 14 với trường hợp đánh giá mà nó tạo ra.

## Vấn đề

Agents vượt qua các bản demo. Họ thất bại trong production theo cách mà các bản demo không thể dự đoán được. Benchmarks trả lời "model này có khả năng rộng rãi không?" chứ không phải "đây có phải là agent shipping bản vá phù hợp cho sản phẩm của tôi không?" Câu trả lời: đánh giá ở ba lớp, chạy liên tục, với mọi guardrail và quy tắc đã học được ánh xạ đến một trường hợp đánh giá.

## Khái niệm

### Ba lớp đánh giá

1. **Static benchmarks** — SWE-bench Verified for code (Bài 19), WebArena/OSWorld để duyệt web / desktop (Bài 20), GAIA cho tổng quát (Lesson 19), BFCL V4 để sử dụng công cụ (Lesson 06). Sử dụng để so sánh chéo model và cổng hồi quy. Ô nhiễm là có thật: SWE-bench+ phát hiện rò rỉ dung dịch 32,67%. Luôn báo cáo điểm đã xác minh / + đã kiểm tra.

2. **Đánh giá ngoại tuyến tùy chỉnh** — hình dạng sản phẩm của bạn:
   - LLM với tư cách là thẩm phán (Langfuse, Phoenix, Opik - Bài 24).
   - Dựa trên thực thi (chạy bản vá, kiểm tra kiểm tra).
   - Dựa trên quỹ đạo (so sánh các chuỗi hành động với vàng; OSWorld-Human cho thấy agents cao nhất 1,4-2,7 lần so với vàng).

3. **Đánh giá trực tuyến** — production:
   - Session phát lại (Langfuse).
   - Cảnh báo được kích hoạt Guardrail (Bài 16, 21).
   - Theo dõi chi phí / độ trễ trên mỗi bước (Bài 23 OTel spans).

### Người đánh giá-optimizer (Anthropic)

Vòng lặp chặt chẽ:

1. Proposer tạo ra đầu ra.
2. Đánh giá thẩm phán.
3. Tinh chỉnh cho đến khi trình đánh giá vượt qua.

Đây là Tự tinh chỉnh (Bài 05) khái quát. Bất kỳ luồng agent nào bạn quan tâm đều có thể được bao bọc trong optimizer đánh giá để đảm bảo độ tin cậy.

### Thực tiễn tốt nhất năm 2026

- Đánh giá trực tiếp bên cạnh mã.
- Chạy trong CI trên mỗi PR.
- Cổng merge trên điểm đánh giá (ví dụ: "không hồi quy > 5% so với chính").
- Mỗi guardrail ánh xạ đến một trường hợp đánh giá.
- Mọi quy tắc đã học (Reflexion, pro-workflow learn-rule) ánh xạ đến một trường hợp thất bại.

### Gắn kết giai đoạn 14 với nhau

Mỗi bài học trong Giai đoạn 14 đều tạo ra các trường hợp đánh giá:

| Bài học | Trường hợp đánh giá mà nó tạo ra |
|--------|------------------------|
| 01 Vòng lặp Agent | Bảo vệ vòng lặp vô hạn, cạn kiệt ngân sách |
| 02 ReWOO | Planner lập kế hoạch lại chính xác khi một công cụ bị lỗi |
| 03 Phản xạ | Phản ánh đã học được áp dụng khi thử lại |
| 05 Self-Refine/CRITIC | Thẩm phán vượt qua đầu ra tinh tế |
| 06 Sử dụng công cụ | Tranh luận cưỡng chế hoạt động; Các công cụ không xác định bị từ chối |
| 07-10 Bộ nhớ | Trích dẫn truy xuất khớp với nguồn; sự thật cũ làm mất hiệu lực |
| 12 Mẫu quy trình làm việc | Mỗi mẫu tạo ra đầu ra chính xác |
| 13 Đồ thị LangGraph | Resume tái tạo trạng thái chính xác |
| 14 diễn viên AutoGen | DLQ bắt các trình xử lý bị hỏng |
| 16 OpenAI Agents SDK | Guardrail chuyến đi trên đầu vào phù hợp |
| 17 Claude Agent SDK | Kết quả Subagent trả về trình điều phối |
| 19-20 Benchmarks | SWE-bench Điểm đã được xác minh, tỷ lệ thành công của WebArena, Hiệu quả của OSorld |
| 21 Sử dụng máy tính | Chốt an toàn mỗi bước được tiêm DOM |
| 23 Hộp đựng | Spans phát ra các thuộc tính bắt buộc |
| 26 Chế độ thất bại | Máy dò gắn thẻ các lỗi đã biết |
| 27 Tiêm Prompt | PVE từ chối thu hồi chất độc |
| 28 Orchestration | Giám sát viên định tuyến đến đúng chuyên gia |
| 29 Runtime hình dạng | DLQ xử lý lỗi N% |

Nếu bộ đánh giá của bạn có các trường hợp cho từng trường hợp, bạn đã bao gồm Giai đoạn 14.

### Nơi phát triển theo định hướng đánh giá thất bại

- **Không có đường cơ sở.** Các đánh giá không có điểm cuối cùng là không thể đọc được. Đường cơ sở của cửa hàng.
- **LLM-thẩm phán không có grounding.** Thẩm phán cũng bị ảo giác. Mô hình CRITIC (Bài 05) — đánh giá cơ sở về các công cụ bên ngoài.
- **Quá phù hợp với đánh giá.** Tối ưu hóa cho đánh giá khác với tính hữu ích của production. Luân phiên các trường hợp.
- **Đánh giá bong tróc.** Các trường hợp không xác định gây ra báo động giả. Ghim hạt giống, trạng thái chụp nhanh.

## Tự xây dựng

`code/main.py` là một harness đánh giá stdlib:

- Trường hợp registry với các danh mục (benchmark, tùy chỉnh, trực tuyến).
- Một agent kịch bản đang được thử nghiệm.
- Vòng lặp optimizer đánh giá: đề xuất, đánh giá, tinh chỉnh cho đến khi đạt hoặc tối đa các vòng.
- Cổng CI: tỷ lệ vượt qua tổng hợp + hồi quy so với đường cơ sở.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: pass/fail mỗi trường hợp, cờ hồi quy CI phán quyết cổng.

## Ứng dụng

- Viết các trường hợp đánh giá cùng repo với mã agent của bạn.
- Chạy chúng trên mọi PR thông qua CI.
- Thất bại trong quá trình xây dựng hồi quy.
- Theo dõi tỷ lệ vượt qua theo thời gian.
- Gắn mọi thất bại production với một trường hợp mới.

## Sản phẩm bàn giao

`outputs/skill-eval-suite.md` xây dựng một bộ đánh giá ba lớp cho một sản phẩm agent với các cổng CI và theo dõi hồi quy.

## Bài tập

1. Lấy một trong những thất bại production của bạn. Viết một trường hợp đánh giá tái tạo nó. Bây giờ agent của bạn có vượt qua nó không?
2. Xây dựng một tiêu chí đánh giá LLM cho miền của bạn với ba khía cạnh (thực tế, giọng điệu, phạm vi). Điểm 50 sessions.
3. Nối bộ đánh giá vào CI. Không thành công trong quá trình xây dựng trên hồi quy >=5%.
4. Thêm số liệu hiệu quả quỹ đạo: agent đã thực hiện bao nhiêu bước so với quỹ đạo vàng?
5. Ánh xạ mọi bài học Giai đoạn 14 với một trường hợp đánh giá trong bộ của bạn. Thiếu thốn? Đó là một khoảng cách cần thu hẹp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| benchmark tĩnh | "Đánh giá có sẵn" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Đánh giá ngoại tuyến tùy chỉnh | "Đánh giá tên miền" | LLM với tư cách là giám khảo / điều hành / quỹ đạo về hình dạng sản phẩm của bạn |
| Đánh giá trực tuyến | "Production đánh giá" | Session phát lại, cảnh báo guardrail cost/latency theo dõi |
| Người đánh giá-optimizer | "Đề xuất-thẩm phán-tinh chỉnh" | Lặp lại cho đến khi giám khảo vượt qua |
| Cổng CI | "Trình chặn Merge" | Thất bại trong quá trình xây dựng hồi quy đánh giá |
| Đường cơ sở | "Last-known-good" | Điểm tham chiếu để phát hiện hồi quy |
| Hiệu quả quỹ đạo | "Bước qua vàng" | Số bước Agent chia cho chuyên gia tối thiểu của con người |

## Đọc thêm

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - "bắt đầu đơn giản, tối ưu hóa với đánh giá"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) - benchmark được tuyển chọn
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) - benchmark sử dụng công cụ
- [Langfuse docs](https://langfuse.com/) - đánh giá + session phát lại trong thực tế
