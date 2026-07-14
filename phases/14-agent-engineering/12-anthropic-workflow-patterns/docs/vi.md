# Các mẫu quy trình làm việc của Anthropic: Đơn giản hơn phức tạp

> Schluntz và Zhang (Anthropic, tháng 12 năm 2024) phân biệt quy trình làm việc (đường dẫn được xác định trước) với agents (sử dụng công cụ động). Năm mẫu quy trình làm việc bao gồm hầu hết các trường hợp. Bắt đầu với các cuộc gọi API trực tiếp. Chỉ thêm agents khi không thể dự đoán được các bước.

**Loại:** Tìm hiểu + Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Agent vòng lặp)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Kể tên năm mẫu quy trình làm việc của Anthropic: chuỗi prompt, định tuyến, song song, điều phối-workers, đánh giá-optimizer.
- Giải thích sự khác biệt giữa agent và quy trình làm việc và chi phí kỹ thuật của từng loại.
- Xác định thời điểm chọn quy trình làm việc thay vì agent (và ngược lại).
- Triển khai tất cả năm mẫu trong stdlib đối với một LLM có kịch bản.

## Vấn đề

Các nhóm tiếp cận nhiều agent frameworks cho các vấn đề muốn có một cuộc gọi chức năng duy nhất. Chi phí là có thật: frameworks thêm các lớp che khuất prompts, che giấu luồng điều khiển và mời gọi sự phức tạp sớm. Bài đăng tháng 12 năm 2024 của Schluntz và Zhang là phản ứng được trích dẫn nhiều nhất trong ngành: bắt đầu đơn giản, chỉ thêm phức tạp khi nó kiếm được chi phí.

## Khái niệm

### Quy trình làm việc so với agents

- **Quy trình làm việc.** LLMs và công cụ được điều phối thông qua các đường dẫn mã được xác định trước. Các kỹ sư sở hữu biểu đồ.
- **Agent.** LLMs tự động điều khiển các công cụ của riêng họ và thực hiện các bước của riêng họ. Người model sở hữu biểu đồ.

Cả hai đều có vị trí của họ. Quy trình làm việc rẻ hơn, nhanh hơn và dễ gỡ lỗi hơn. Agents mở khóa các vấn đề kết thúc mở nhưng làm cho các chế độ thất bại khó suy luận hơn.

### Các LLM tăng cường

Nền tảng cho cả năm mẫu: một LLM với ba khả năng được kết nối - tìm kiếm (truy xuất), công cụ (hành động), bộ nhớ (persistence). Bất kỳ cuộc gọi API nào cũng có thể sử dụng những thứ này.

### Năm mẫu

1. **Prompt chuỗi.** Đầu ra của cuộc gọi 1 là đầu vào cho cuộc gọi 2. Sử dụng khi một tác vụ có sự phân tách tuyến tính rõ ràng. Cổng lập trình tùy chọn giữa các bước.

2. **Định tuyến.** Bộ phân loại LLM chọn LLM hoặc công cụ xuôi dòng nào để gọi. Sử dụng khi các đầu vào khác nhau cần xử lý khác nhau (hỗ trợ cấp 1 so với hoàn tiền so với lỗi và doanh số).

3. **Song song.** Chạy đồng thời N LLM cuộc gọi, tổng hợp kết quả. Hai hình dạng: phân chia (các khối khác nhau) và bỏ phiếu (cùng prompt, N chạy, majority/synthesis).

4. **Orchestrator-workers.** Một orchestrator LLM tự động quyết định workers nào (cũng LLMs) để chạy và tổng hợp đầu ra của chúng. Tương tự như agent vòng lặp nhưng trình điều phối không lặp lại vô thời hạn.

5. **Người đánh giá-optimizer.** Một LLM đề xuất một câu trả lời, một LLM khác đánh giá nó. Lặp lại cho đến khi trình đánh giá vượt qua. Đây là Tự tinh chỉnh (Bài 05) khái quát.

### Nơi quy trình làm việc đánh bại agents

- **Nhiệm vụ có thể dự đoán.** Nếu bạn có thể liệt kê các bước, bạn nên làm.
- **Nhiệm vụ có giới hạn chi phí.** Quy trình làm việc có giới hạn số bước; agents có thể xoắn ốc.
- **Nhiệm vụ ràng buộc tuân thủ.** Kiểm toán viên muốn đọc biểu đồ chứ không phải suy ra từ quỹ đạo.

### Nơi agents đánh bại quy trình làm việc

- **Nghiên cứu mở.** Khi nào bước tiếp theo phụ thuộc vào những gì bước cuối cùng trả về.
- **Nhiệm vụ có độ dài thay đổi.** Số phút đến giờ làm việc không xác định số bước.
- **Miền mới.** Khi bạn chưa biết quy trình làm việc phù hợp - khám phá trước, hệ thống hóa sau.

### Người bạn đồng hành về kỹ thuật ngữ cảnh

"Kỹ thuật ngữ cảnh hiệu quả cho AI agents" (Anthropic 2025) chính thức hóa kỷ luật liền kề: cửa sổ 200k là ngân sách, không phải container. Bao gồm những gì, khi nào nên thu gọn, khi nào để ngữ cảnh phát triển. Được đề cập chi tiết trong bài học Giai đoạn 14 về nén ngữ cảnh (Giai đoạn 14, bài 06 trước đó trong chương trình giảng dạy này trước khi đánh số lại).

## Tự xây dựng

`code/main.py` triển khai tất cả năm mẫu quy trình làm việc đối với một `ScriptedLLM`:

- `prompt_chain(input, steps)` — tuần tự.
- `route(input, classifier, handlers)` - phân loại + công văn.
- `parallel_vote(prompt, n, aggregator)` - N chạy, tổng hợp.
- `orchestrator_workers(task, workers)` - người điều phối chọn workers.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` - vòng lặp cho đến khi vượt qua.

Chạy nó:

```
python3 code/main.py
```

Mỗi mẫu in trace của nó. Tổng số dòng mã trên mỗi mẫu là ~10-15; Chi phí của một framework được tính bằng hàng nghìn.

## Ứng dụng

- Cuộc gọi API trực tiếp cho hầu hết các nhiệm vụ.
- Chỉ Framework khi mẫu thực sự cần trạng thái bền bỉ (LangGraph), đồng thời diễn viên model (AutoGen v0.4) hoặc tạo mẫu vai trò (CrewAI).
- Tiếp cận Claude Agent SDK khi bạn muốn Claude Code harness hình dạng mà không cần xây dựng lại.

## Sản phẩm bàn giao

`outputs/skill-workflow-picker.md` chọn mẫu phù hợp cho mô tả nhiệm vụ nhất định, bao gồm lý do quyết định và đường dẫn tái cấu trúc đến agent nếu quy trình làm việc không thành công.

## Bài tập

1. Triển khai định tuyến với ngưỡng tin cậy. Dưới ngưỡng -> leo thang thành con người. Ngưỡng cho trường hợp sử dụng hỗ trợ cấp 1 ở đâu?
2. Thêm một timeout vào `parallel_vote`. Điều gì xảy ra khi một cuộc gọi bị treo? Làm thế nào để bạn tổng hợp với các phiếu bầu bị thiếu?
3. Biến `evaluator_optimizer` thành một tên cướp: giữ 2 đầu ra hàng đầu qua các lần lặp lại để kết quả tốt muộn không bị ghi đè bởi kết quả xấu muộn.
4. Kết hợp chuỗi prompt với định tuyến: bộ định tuyến chọn một trong ba chuỗi. Đo lường chi phí token so với một giải pháp thay thế prompt lớn duy nhất.
5. Chọn một trong các production features của bạn. Vẽ biểu đồ quy trình làm việc. Đếm bước. Một agent có thực sự tốt hơn ở đây không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Quy trình làm việc | "Quy trình được xác định trước" | Biểu đồ LLM và lệnh gọi công cụ thuộc sở hữu của kỹ sư |
| Agent | "AI tự trị" | đồ thị thuộc sở hữu của Model; Hướng công cụ động |
| Tăng cường LLM | "LLM với các công cụ" | LLM + tìm kiếm + công cụ + bộ nhớ; Đơn vị nguyên tử |
| Prompt chuỗi | "Cuộc gọi tuần tự" | Đầu ra của cuộc gọi N là đầu vào để gọi N + 1 |
| Định tuyến | "Công văn phân loại" | Chọn chain/model nào xử lý đầu vào |
| Song song hóa | "Quạt ra" | N cuộc gọi đồng thời; Tổng hợp bằng cách phân chia hoặc bỏ phiếu |
| Orchestrator-workers | "Điều phối viên agent" | Orchestrator LLM chọn chuyên gia LLMs động |
| Người đánh giá-optimizer | "Người đề xuất + thẩm phán" | Lặp lại cho đến khi trình đánh giá vượt qua; Tự tinh chỉnh tổng quát |

## Đọc thêm

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — năm mẫu quy trình làm việc
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — kỷ luật đồng hành
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — khi biểu đồ có trạng thái kiếm được chi phí
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — mô hình workers của người điều phối, được sản xuất hóa
