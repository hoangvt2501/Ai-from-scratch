# Benchmarks: SWE-bench, GAIA, AgentBench

> Ba benchmarks neo agent đánh giá vào năm 2026. SWE-bench kiểm tra việc vá mã. GAIA kiểm tra việc sử dụng công cụ tổng quát. AgentBench kiểm tra suy luận đa môi trường. Biết thành phần của chúng, câu chuyện ô nhiễm của chúng và những gì chúng không đo lường.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 06 (Sử dụng công cụ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Đặt tên cho harness kiểm tra của SWE-bench (FAIL_TO_PASS) và giải thích lý do tại sao nó kiểm tra đơn vị.
- Giải thích lý do tại sao SWE-bench Verified (OpenAI, 500 tác vụ) tồn tại và những gì nó loại bỏ.
- Mô tả thiết kế của GAIA: đơn giản cho con người, khó cho AI; ba mức độ khó.
- Đặt tên cho tám môi trường của AgentBench và trình chặn chính của nó cho LLMs mã nguồn mở.
- Tóm tắt phát hiện ô nhiễm SWE-bench+ và ý nghĩa của nó.

## Vấn đề

Bảng xếp hạng cho bạn biết model nào thắng trong một benchmark. Họ không cho bạn biết:

- benchmark có bị nhiễm bẩn hay không (giải pháp trong dữ liệu training, rò rỉ thử nghiệm).
- Liệu benchmark có đo lường những gì bạn quan tâm hay không (mã so với duyệt web so với tổng quát).
- Liệu trình đánh giá có mạnh mẽ hay không (đối sánh AST, kiểm tra trạng thái, đánh giá của con người).

Biết ba benchmarks neo và chế độ lỗi của chúng trước khi bạn trích dẫn một con số.

## Khái niệm

### SWE-bench (Jimenez và cộng sự, ICLR 2024 miệng)

- 2.294 số GitHub thực tế từ 12 Python repos nổi tiếng.
- Agent nhận được: cơ sở mã ở commit tiền tố + mô tả vấn đề ngôn ngữ tự nhiên.
- Agent sản xuất: một bản vá.
- Người đánh giá: áp dụng bản vá, chạy bộ kiểm tra của repo. Bản vá phải lật FAIL_TO_PASS các bài kiểm tra (trước đây thất bại, bây giờ đạt) mà không phá vỡ PASS_TO_PASS kiểm tra.

SWE-agent (Yang et al., 2024) đạt 12,5% khi phát hành bằng cách nhấn mạnh giao diện agent-máy tính (lệnh trình chỉnh sửa tệp, cú pháp tìm kiếm mà model hiểu).

### SWE-bench đã được xác minh

OpenAI, tháng 8 năm 2024. Tập hợp con 500 nhiệm vụ do con người quản lý. Loại bỏ các vấn đề mơ hồ, kiểm tra không đáng tin cậy và các tác vụ mà bản sửa lỗi không rõ ràng. benchmark chính cho "agent ship của bạn có bản vá thật không?"

### Ô nhiễm

- Hơn 94% các vấn đề của SWE-bench xảy ra trước hầu hết các model giới hạn.
- **SWE-bench+** tìm thấy 32,67% các bản vá lỗi thành công bị rò rỉ giải pháp trong văn bản sự cố (model thấy bản sửa lỗi trong mô tả) và 31,08% nghi ngờ do phạm vi kiểm tra yếu.
- Đã xác minh sạch hơn nhưng không bị nhiễm bẩn.

Ý nghĩa thực tế: một model đạt 50% trên băng ghế dự bị có thể đạt 35% trên SWE-bench +. Luôn báo cáo cả hai nếu bạn tuyên bố hiệu suất SWE-bench.

### GAIA (Mialon và cộng sự, tháng 11 năm 2023)

- 466 câu hỏi; 300 được giữ lại cho bảng xếp hạng riêng tại huggingface.co/gaia-benchmark.
- Triết lý thiết kế: "khái niệm đơn giản đối với con người (92%) nhưng khó đối với AI (GPT-4 với plugin: 15%)."
- Kiểm tra lý luận, đa phương thức, web, sử dụng công cụ.
- Ba mức độ khó; Cấp độ 3 yêu cầu chuỗi công cụ dài trên các phương thức.

GAIA là những gì bạn chạy để đo lường "khả năng tổng quát". Đừng nhầm lẫn với benchmarks cụ thể của mã.

### AgentBench (Liu và cộng sự, ICLR 2024)

- 8 môi trường trên mã (Bash, DB, KG), trò chơi (Alfworld, LTP), web (WebShop, Mind2Web) và thế hệ mở.
- Nhiều lượt, ~4k-13k lượt mỗi tách.
- Phát hiện chính: lý luận dài hạn, ra quyết định và hướng dẫn sau đây là những yếu tố cản trở OSS LLMs bắt kịp thương mại.

### Những gì chúng không đo lường

- Chi phí hoạt động trong thế giới thực (tokens, đồng hồ treo tường).
- Hành vi an toàn trong điều kiện bất lợi.
- Hiệu suất trên miền của bạn (sử dụng đánh giá của riêng bạn, Bài 30).
- Hỏng đuôi (benchmarks trung bình; production nhà khai thác quan tâm đến 1% tồi tệ nhất.

### Điểm chuẩn bị lỗi ở đâu

- **Cố định một số.** SWE-bench 50% cho bạn biết ít hơn chi phí P50/P75/P95 + phân phối bước.
- **Tuyên bố bị ô nhiễm.** Báo cáo SWE-bench mà không đề cập đến Đã xác minh hoặc SWE-bench+ là gây hiểu lầm.
- **Benchmark-as-development-target.** Tối ưu hóa cho benchmark khác với tính hữu ích của production.

## Tự xây dựng

`code/main.py` triển khai một harness giống như băng ghế dự bị SWE:

- Tác vụ sửa lỗi tổng hợp (3 nhiệm vụ).
- Một "agent" theo kịch bản đề xuất các bản vá.
- Một trình chạy thử nghiệm kiểm tra FAIL_TO_PASS (lỗi hiện đã được sửa) và PASS_TO_PASS (không có gì bị hỏng).
- Một bộ phân loại độ khó kiểu GAIA dựa trên độ sâu phân tích câu hỏi.

Chạy nó:

```
python3 code/main.py
```

Đầu ra hiển thị tốc độ giải quyết cho mỗi nhiệm vụ + mỗi độ khó và làm cho các quy tắc đánh giá cụ thể.

## Ứng dụng

- **SWE-bench Verified** cho mã agents. Luôn báo cáo điểm đã xác minh.
- **GAIA** cho các agents tổng quát. Sử dụng phân chia bảng xếp hạng riêng tư.
- **AgentBench** để so sánh nhiều môi trường.
- **Đánh giá tùy chỉnh** (Bài 30) cho hình dạng thực tế của sản phẩm của bạn.

## Sản phẩm bàn giao

`outputs/skill-benchmark-harness.md` xây dựng một harness kiểu SWE-bench cho bất kỳ cặp tác vụ cơ sở mã nào có cổng FAIL_TO_PASS / PASS_TO_PASS.

## Bài tập

1. Chuyển harness đồ chơi để chạy trên repo thực (chọn một trong số của bạn). Viết 3 bài kiểm tra FAIL_TO_PASS cho các lỗi đã biết.
2. Thêm chỉ số số bước. Trên 3 nhiệm vụ của bạn, có bao nhiêu bước agent cho mỗi độ phân giải?
3. Đọc bài SWE-bench +. Triển khai kiểm tra rò rỉ giải pháp (khớp mẫu văn bản vấn đề với sự khác biệt).
4. Tải xuống câu hỏi GAIA từ phần chia công khai. Trace những gì một class agent GPT-4 sẽ làm. Nó cần những công cụ nào?
5. Đọc bảng phân tích theo từng môi trường của AgentBench. Môi trường nào phản ánh bề mặt sản phẩm của bạn? "SOTA" trông như thế nào ở đó?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| SWE-băng ghế | "Mã agent benchmark" | 2.294 vấn đề GitHub; bản vá phải lật FAIL_TO_PASS kiểm tra |
| SWE-bench đã được xác minh | "Băng ghế SWE sạch" | 500 nhiệm vụ do con người quản lý, OpenAI |
| FAIL_TO_PASS | "Sửa cổng" | Các bài kiểm tra trước đây không thành công phải vượt qua sau bản vá |
| PASS_TO_PASS | "Cổng không hồi quy" | Các bài kiểm tra đã vượt qua và vẫn phải vượt qua |
| GAIA | "Tổng quát benchmark" | 466 câu hỏi đa công cụ dễ / AI khó |
| Băng ghế đại lý | "Đa môi trường benchmark" | 8 môi trường; nhiều lượt đường chân trời dài |
| Ô nhiễm | "Rò rỉ Training bộ" | Benchmark nhiệm vụ hiện diện trong model training |
| SWE-băng ghế + | "Kiểm toán ô nhiễm" | 32,67% rò rỉ dung dịch được tìm thấy trong các miếng dán SWE-bench thành công |

## Đọc thêm

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — benchmark ban đầu
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — tập hợp con được tuyển chọn
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — benchmark tổng quát
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — Bộ đa môi trường
