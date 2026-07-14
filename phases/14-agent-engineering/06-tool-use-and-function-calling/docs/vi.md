# Sử dụng và Function Calling dụng cụ

> Toolformer (Schick và cộng sự, 2023) bắt đầu chú thích công cụ tự giám sát. Bảng xếp hạng Berkeley Function Calling V4 (Patil và cộng sự, 2025) đặt ra tiêu chuẩn năm 2026: 40% agentic, 30% nhiều lượt, 10% trực tiếp, 10% không trực tiếp, 10% ảo giác. Một lượt được giải quyết. Bộ nhớ, ra quyết định năng động và chuỗi công cụ dài hạn thì không.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent), Giai đoạn 13 · 01 (Function Calling Tìm hiểu sâu)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Giải thích tín hiệu training tự giám sát của Toolformer: chỉ giữ chú thích công cụ khi quá trình thực thi giảm token loss tiếp theo.
- Kể tên năm hạng mục đánh giá của BFCL V4 và những gì mỗi hạng mục đo lường.
- Triển khai công cụ stdlib registry với xác thực schema, cưỡng chế đối số và sandbox thực thi.
- Chẩn đoán ba vấn đề mở năm 2026: chuỗi công cụ đường chân trời dài, ra quyết định năng động và bộ nhớ.

## Vấn đề

Việc sử dụng công cụ ban đầu được hỏi: model có thể dự đoán một cuộc gọi hàm chính xác không? Việc sử dụng công cụ hiện đại đặt ra câu hỏi: liệu model có thể xâu chuỗi các công cụ qua 40 bước, với bộ nhớ, với observability một phần, với sự phục hồi từ lỗi công cụ, mà không có các công cụ ảo giác không tồn tại?

Toolformer đã thiết lập đường cơ sở: models có thể học khi nào nên gọi các công cụ với tính năng tự giám sát. BFCL V4 xác định mục tiêu đánh giá năm 2026. Khoảng cách giữa chúng là không gian production agents sống.

## Khái niệm

### Toolformer (Schick và cộng sự, NeurIPS 2023)

Ý tưởng: hãy để model chú thích kho dữ liệu pretraining của riêng mình với các cuộc gọi API ứng viên. Đối với mỗi ứng cử viên, hãy thực hiện nó. Chỉ giữ chú thích nếu bao gồm kết quả công cụ làm giảm loss trong token tiếp theo. Fine-tune trên kho dữ liệu đã lọc.

Các công cụ được bảo hiểm: máy tính, hệ thống QA, công cụ tìm kiếm, dịch giả, lịch. Tín hiệu tự giám sát hoàn toàn là về việc liệu công cụ có giúp dự đoán văn bản hay không - không có nhãn của con người.

Tỷ lệ kết quả: việc sử dụng công cụ xuất hiện trên quy mô lớn. models nhỏ hơn bị tổn thương do chú thích công cụ; lợi nhuận models lớn hơn. Đây là lý do tại sao models biên giới năm 2026 có việc sử dụng công cụ mạnh mẽ trong khi hầu hết các models 7B cần fine-tuning sử dụng công cụ rõ ràng để đáng tin cậy.

### Bảng xếp hạng Berkeley Function Calling V4 (Patil và cộng sự, ICML 2025)

BFCL là đánh giá thực tế năm 2026. Thành phần V4:

- **Agentic (40%)** — quỹ đạo agent đầy đủ: trí nhớ, nhiều lượt, quyết định năng động.
- **Multi-Turn (30%)** — cuộc trò chuyện tương tác với chuỗi công cụ.
- **Trực tiếp (10%)** — prompts thực do người dùng gửi (phân phối khó hơn).
- **Non-Live (10%)** — các trường hợp thử nghiệm tổng hợp.
- **Ảo giác (10%)** — phát hiện khi không có công cụ nào được gọi.

V3 giới thiệu đánh giá dựa trên trạng thái: sau một trình tự công cụ, hãy kiểm tra trạng thái thực tế của API (ví dụ: "tệp đã được tạo chưa?") thay vì khớp với AST của các lệnh gọi công cụ. V4 đã thêm các danh mục tìm kiếm web, bộ nhớ và độ nhạy định dạng.

Phát hiện quan trọng năm 2026: function calling một lượt gần như đã được giải quyết. Lỗi tập trung vào bộ nhớ (mang bối cảnh qua các lượt), ra quyết định năng động (chọn công cụ dựa trên kết quả prior), chuỗi đường chân trời dài (trôi sau 20+ bước) và phát hiện ảo giác (từ chối gọi khi không có công cụ nào phù hợp).

### Công cụ schema

Mỗi nhà cung cấp đều có một schema. Chúng khác nhau về chi tiết nhưng có cùng hình dạng:

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic sử dụng `input_schema` trực tiếp. OpenAI sử dụng `function.parameters`. Cả hai đều chấp nhận JSON Schema. Mô tả có khả năng chịu tải - model đọc chúng để chọn công cụ phù hợp. Mô tả công cụ xấu là nguyên nhân gốc rễ #1 của lỗi chọn sai công cụ.

### Xác thực đối số

Không tin tưởng cuộc gọi công cụ. Xác thực:

1. **Loại cưỡng chế.** Model có thể trả về một chuỗi "5" trong đó schema nói int. Ép buộc nếu rõ ràng; từ chối nếu không.
2. **Xác thực enum.** Nếu schema cho biết `status in {"open", "closed"}` và model phát ra `"in_progress"`, hãy từ chối với lỗi mô tả.
3. **Các trường bắt buộc.** Thiếu trường bắt buộc -> quan sát lỗi ngay lập tức trở lại model, không phải là sự cố.
4. **Xác thực định dạng.** Ngày, email, URL — xác thực bằng trình phân tích cú pháp cụ thể, không phải biểu thức chính quy.

Mỗi lần xác thực thất bại sẽ trả về một quan sát có cấu trúc để model có thể thử lại với hình dạng chính xác.

### Lệnh gọi công cụ song song

Các nhà cung cấp hiện đại hỗ trợ các cuộc gọi công cụ song song trong một lượt trợ lý. Vòng lặp:

1. Model phát ra 3 lệnh gọi công cụ với các `tool_use_id` riêng biệt.
2. Runtime thực thi chúng (song song nếu độc lập).
3. Mỗi kết quả quay trở lại dưới dạng một khối `tool_result` tương quan với `tool_use_id`.

Quy tắc kỹ thuật: coi ID tương quan là chịu tải. Hoán đổi chúng và bạn nhận được định tuyến sai công cụ thành kết quả sai.

### Sandbox

Thực thi công cụ là ranh giới sandbox. Xem Bài 09 để biết chi tiết. Phiên bản ngắn: mọi công cụ nên chỉ định read/write bề mặt, truy cập mạng, timeout, giới hạn bộ nhớ. `run_shell(cmd)` chung là một lá cờ đỏ; `git_status()` cụ thể an toàn hơn.

```figure
tool-routing
```

## Tự xây dựng

`code/main.py` triển khai một công cụ hình production registry:

- JSON Schema trình xác thực tập con (chỉ stdlib).
- Đăng ký công cụ với mô tả, schema nhập, timeout và trình thực thi.
- Cưỡng chế lập luận và xác thực enum.
- Gửi công cụ song song với ID tương quan.
- Quan sát lỗi dưới dạng chuỗi có cấu trúc.

Chạy nó:

```
python3 code/main.py
```

trace cho thấy một agent nhỏ gọi ba công cụ trong một lượt, với một cuộc gọi cố tình sai định dạng bị từ chối với lỗi mô tả mà model có thể hành động.

## Ứng dụng

Mỗi nhà cung cấp đều có schema công cụ riêng — Anthropic, OpenAI, Gemini, Bedrock. Sử dụng lớp dịch (OpenAI Agents SDK, Vercel AI SDK, bộ chuyển đổi công cụ LangChain) nếu bạn cần nhiều nhà cung cấp. BFCL là benchmark tham chiếu - chạy nó với agent của bạn trước khi shipping nếu việc sử dụng công cụ là trọng tâm của sản phẩm.

## Sản phẩm bàn giao

`outputs/skill-tool-registry.md` tạo danh mục công cụ, schema và registry cho một miền tác vụ nhất định. Bao gồm kiểm tra chất lượng mô tả (mô tả của mỗi công cụ có cho model biết khi nào nên sử dụng không?).

## Bài tập

1. Thêm một công cụ "no-op" cho phép model từ chối sử dụng bất kỳ công cụ nào khác một cách rõ ràng. Đo trên bài kiểm tra ảo giác giống BFCL.
2. Thực hiện cưỡng chế đối số cho int-as-string và float-as-string. Cưỡng ép bắt đầu che giấu lỗi thực sự ở đâu?
3. Thêm một timeout cho mỗi dụng cụ và một bộ ngắt mạch (từ chối công cụ trong 60 giây sau 3 lần hỏng hóc liên tiếp). Điều này thay đổi gì về cách model phục hồi?
4. Đọc mô tả BFCL V4. Chọn một danh mục (ví dụ: "nhiều lượt") và chạy 10 ví dụ prompts qua agent của bạn. Tỷ lệ vượt qua báo cáo.
5. Chuyển trình xác thực stdlib sang Pydantic hoặc Zod. Pydantic/Zod đã bắt được điều gì mà đồ chơi đã bỏ lỡ?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Function calling | "Sử dụng công cụ" | Lệnh gọi công cụ đầu ra có cấu trúc với schema đã được xác thực |
| Máy tạo công cụ | "Chú thích công cụ tự giám sát" | Schick 2023 — giữ các lệnh gọi công cụ có kết quả giảm token loss tiếp theo |
| BFCL | "Bảng xếp hạng Berkeley Function Calling" | benchmark 2026: 40% agentic, 30% nhiều lượt, 10% trực tiếp, 10% không trực tiếp, 10% ảo giác |
| Công cụ schema | "Chữ ký chức năng cho model" | tên, mô tả JSON Schema đối số |
| tool_use_id | "ID tương quan" | Gắn lệnh gọi công cụ với kết quả của nó; Cần thiết cho công văn song song |
| Phát hiện ảo giác | "Biết khi nào không nên gọi" | Danh mục V4: từ chối gọi khi không có công cụ phù hợp |
| Cưỡng chế lập luận | "Sửa chữa từ chuỗi đến int" | Các bản sửa lỗi hẹp cho schema-không khớp có thể dự đoán được; từ chối nếu mơ hồ |
| Sandbox | "Ranh giới thực thi công cụ" | Mỗi công cụ read/write bề mặt, mạng, timeout, nắp bộ nhớ |

## Đọc thêm

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — chú thích công cụ tự giám sát
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) - benchmark đánh giá năm 2026
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) - production công cụ schema trong Claude Agent SDK
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — loại công cụ chức năng và Guardrails
