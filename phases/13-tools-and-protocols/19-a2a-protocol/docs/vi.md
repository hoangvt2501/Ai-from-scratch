# A2A — Giao thức Agent-Agent

> MCP là công cụ agent. A2A (Agent2Agent) là agent-to-agent - một giao thức mở để cho phép các agents mờ đục được xây dựng trên các frameworks khác nhau cộng tác. Được Google phát hành vào tháng 4 năm 2025, quyên góp cho Linux Foundation vào tháng 6 năm 2025, đạt phiên bản 1.0 vào tháng 4 năm 2026 với 150+ người ủng hộ bao gồm AWS, Cisco, Microsoft, Salesforce, SAP và ServiceNow. Nó đã hấp thụ ACP của IBM và thêm phần mở rộng thanh toán AP2. Bài học này đi qua Thẻ Agent, Vòng đời tác vụ và hai ràng buộc transport.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, Agent Card + Task harness)
**Kiến thức tiên quyết:** Giai đoạn 13 · 06 (MCP nguyên tắc cơ bản), Giai đoạn 13 · 08 (MCP khách hàng)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Phân biệt các trường hợp sử dụng agent-to-tool (MCP) với các trường hợp sử dụng agent-to-agent (A2A).
- Xuất bản Thẻ Agent tại `/.well-known/agent.json` với siêu dữ liệu skills và endpoint.
- Đi theo vòng đời Tác vụ (được gửi → làm việc → yêu cầu đầu vào → hoàn thành / không thành công / bị hủy / bị từ chối).
- Sử dụng Tin nhắn có các bộ phận (văn bản, tệp, dữ liệu) và Artifacts làm đầu ra.

## Vấn đề

Một agent dịch vụ khách hàng cần ủy quyền viết báo cáo cho một agent viết chuyên ngành. Các tùy chọn trước khi A2A:

- REST API tùy chỉnh. Hoạt động nhưng mỗi cặp đều là một lần.
- Cơ sở mã được chia sẻ. Yêu cầu hai agents chạy cùng một framework.
- MCP. Không phù hợp: MCP dành cho các công cụ gọi điện, không phải dành cho hai agents cộng tác trong khi vẫn giữ nguyên lý luận nội bộ không rõ ràng của mỗi agent.

A2A lấp đầy khoảng trống. Nó models tương tác như một agent gửi Nhiệm vụ cho một Nhiệm vụ khác, với vòng đời, tin nhắn và artifacts. Trạng thái bên trong của agent được gọi vẫn mờ đục - người gọi chỉ thấy chuyển đổi trạng thái tác vụ và kết quả cuối cùng.

A2A là giao thức "để agents qua frameworks nói chuyện với nhau". Nó không thay thế MCP; Cả hai bổ sung cho nhau.

## Khái niệm

### Thẻ Agent

Mỗi agent tuân thủ A2A sẽ xuất bản một thẻ tại `/.well-known/agent.json`:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Khám phá dựa trên URL: tìm nạp thẻ, tìm hiểu URL của A2A endpoint, liệt kê skills.

### Thẻ Agent có chữ ký (AP2)

Tiện ích mở rộng AP2 (Tháng 9 năm 2025) thêm chữ ký mật mã vào Thẻ Agent. Một nhà xuất bản ký vào thẻ của riêng mình với JWT; người tiêu dùng xác minh. Ngăn chặn hành vi mạo danh.

### Vòng đời tác vụ

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Khách hàng bắt đầu với `tasks/send`. Các agent được gọi chuyển đổi qua các trạng thái; Khách hàng đăng ký cập nhật trạng thái thông qua SSE hoặc thăm dò ý kiến.

### Tin nhắn và các bộ phận

Một tin nhắn mang một hoặc nhiều Phần:

- `text` — nội dung đơn giản.
- `file` — đốm màu base64 với mimeType.
- `data` - JSON payload được nhập (đầu vào có cấu trúc cho agent được gọi).

Ví dụ:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Đầu ra là Artifacts, không phải chuỗi thô. Một Artifact là một đầu ra được đặt tên, được nhập:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts có thể được phát trực tuyến dưới dạng khối. Người gọi tích lũy.

### Hai ràng buộc transport

1. **JSON-RPC qua HTTP.** `/a2a` endpoint, POST cho các yêu cầu, SSE tùy chọn cho streaming. Ràng buộc mặc định.
2. **gRPC.** Dành cho môi trường doanh nghiệp nơi gRPC là gốc.

Cả hai ràng buộc đều mang cùng một hình dạng thông điệp logic.

### Bảo quản độ mờ

Một nguyên tắc thiết kế quan trọng: trạng thái bên trong của agent được gọi là mờ đục. Người gọi nhìn thấy trạng thái tác vụ và artifacts. Người được gọi là chain-of-thought của agent, các cuộc gọi công cụ của nó, ủy quyền agent phụ của nó - tất cả đều vô hình. Điều này khác với MCP, nơi các lệnh gọi công cụ minh bạch.

Cơ sở lý luận: A2A cho phép các đối thủ cạnh tranh cộng tác mà không tiết lộ nội bộ. A2A có thể là "gọi dịch vụ khách hàng này agent" mà người gọi không tìm hiểu cách agent đó triển khai dịch vụ.

### Dòng thời gian

- **2025-04-09.** Google công bố A2A.
- **2025-06-23.** Quyên góp cho Linux Foundation.
- **2025-08.** Hấp thụ ACP của IBM.
- **2025-09.** ships Gia hạn AP2 (Thanh toán Agent).
- **2026-04.** Phiên bản 1.0 được phát hành với 150+ tổ chức hỗ trợ.

### Mối quan hệ với MCP

| Kích thước | MCP | A2A |
|-----------|-----|-----|
| Trường hợp sử dụng | Agent dụng cụ | Agent agent |
| Độ mờ | Lệnh gọi công cụ minh bạch | Lý luận bên trong mờ đục |
| Người gọi điển hình | Agent runtime | Một agent khác |
| Tiểu bang | Kết quả gọi công cụ | Nhiệm vụ có vòng đời |
| Authorization | OAuth 2.1 (Giai đoạn 13 · 16) | Thẻ Agent có chữ ký JWT (AP2) |
| Transport | Stdio / HTTP có thể phát trực tuyến | JSON-RPC trên HTTP / gRPC |

Sử dụng MCP khi bạn muốn gọi một công cụ cụ thể. Sử dụng A2A khi bạn muốn ủy thác toàn bộ nhiệm vụ cho một agent khác. Nhiều hệ thống production sử dụng cả hai: agent sử dụng MCP cho lớp công cụ và A2A cho lớp cộng tác.

## Ứng dụng

`code/main.py` thực hiện một A2A harness tối thiểu: một agent nghiên cứu xuất bản thẻ của mình, một nhà văn agent nhận được một `tasks/send` với các phần bao gồm PDF và hướng dẫn văn bản, chuyển đổi thông qua làm việc → input_required → làm việc → hoàn thành và trả về một artifact văn bản. Tất cả các stdlib; sử dụng transport trong bộ nhớ để tập trung vào hình dạng thư.

Những gì cần xem:

- Agent Hình dạng JSON thẻ.
- Gán id tác vụ và chuyển đổi trạng thái.
- Tin nhắn có các bộ phận hỗn hợp.
- Yêu cầu đầu vào branch nhiệm vụ giữa.
- Artifact trả lại sau khi hoàn thành.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-a2a-agent-spec.md`. Với một agent mới có thể được gọi bởi các agents khác, skill sẽ tạo ra bản thiết kế JSON, skills schema và endpoint Thẻ Agent.

## Bài tập

1. Chạy `code/main.py`. Trace toàn bộ vòng đời Tác vụ, bao gồm cả tạm dừng yêu cầu đầu vào trong đó agent được gọi yêu cầu làm rõ.

2. Thêm Thẻ Agent đã ký. Ký bằng HMAC trên JSON chính tắc của thẻ. Viết trình xác minh và xác nhận nó không thành công trên thẻ đột biến.

3. Thực hiện streaming nhiệm vụ: người viết agent phát ra ba đoạn artifact gia tăng trên SSE và người gọi tích lũy chúng.

4. Thiết kế một A2A agent bao bọc một MCP server. Ánh xạ từng công cụ MCP thành một A2A skill. Lưu ý sự đánh đổi - độ mờ nào bị mất?

5. Đọc thông báo A2A v1.0 và xác định một feature chưa được triển khai bởi bất kỳ framework nào kể từ tháng 4 năm 2026. (Gợi ý: nó liên quan đến ủy quyền nhiệm vụ nhiều bước nhảy.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| A2A | "Giao thức Agent-to-Agent" | Giao thức mở để cộng tác agent không rõ ràng |
| Thẻ Agent | "`.well-known/agent.json`" | Siêu dữ liệu đã xuất bản mô tả skills và endpoint của agent |
| Skill | "Một đơn vị có thể gọi" | Một hoạt động được đặt tên mà agent hỗ trợ (công cụ tương tự sang MCP) |
| Nhiệm vụ | "Đơn vị đoàn" | Mục công việc có vòng đời và artifact cuối cùng |
| Thông điệp | "Nhập nhiệm vụ" | Mang các bộ phận (văn bản, tệp, dữ liệu) |
| Phần | "Khối được đánh máy" | `text` / `file` / `data` yếu tố của tin nhắn |
| Artifact | "Đầu ra nhiệm vụ" | Đầu ra được đặt tên, nhập trả về khi hoàn thành |
| AP2 | "Giao thức thanh toán Agent" | Gia hạn Thẻ Agent đã ký để ủy thác và thanh toán |
| Độ mờ | "Hợp tác hộp đen" | Được gọi là nội bộ của agent bị ẩn khỏi người gọi |
| Yêu cầu đầu vào | "Tạm dừng nhiệm vụ" | Trạng thái vòng đời khi agent cần thêm thông tin |

## Đọc thêm

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — đặc điểm kỹ thuật A2A chuẩn
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — triển khai tham khảo và SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) - Chuyển giao quản trị tháng 6 năm 2025
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — lộ trình và động lực đối tác
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — ghi chú phát hành v1.0 và hướng dẫn tương thích ngược
