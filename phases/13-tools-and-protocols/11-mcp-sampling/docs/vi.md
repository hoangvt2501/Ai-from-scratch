# MCP Sampling — Hoàn thành LLM yêu cầu Server và vòng lặp Agent

> Hầu hết MCP servers là trình thực thi ngu ngốc: lấy đối số, chạy mã, trả về nội dung. Sampling cho phép một server lật hướng: nó yêu cầu LLM của khách hàng đưa ra quyết định. Điều này cho phép các vòng lặp agent được lưu trữ server mà không cần server sở hữu bất kỳ thông tin đăng nhập model nào. SEP-1577, merged vào ngày 25 tháng 11 năm 2025, đã thêm các công cụ bên trong các yêu cầu sampling để vòng lặp có thể bao gồm suy luận sâu hơn. Lưu ý về rủi ro trôi dạt: hình dạng công cụ trong sampling SEP-1577 đã được thử nghiệm cho đến quý 1 năm 2026 và vẫn đang ổn định trong SDK APIs.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, sampling harness)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server), Giai đoạn 13 · 10 (Tài nguyên và prompts)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích những gì `sampling/createMessage` giải quyết (vòng lặp lưu trữ server không có khóa API bên server).
- Triển khai một server yêu cầu khách hàng lấy mẫu qua prompt nhiều lượt và trả về kết quả hoàn thành.
- Sử dụng `modelPreferences` (ưu tiên chi phí / tốc độ / thông minh) để hướng dẫn lựa chọn model khách hàng.
- Xây dựng một công cụ `summarize_repo` lặp lại nội bộ thông qua sampling thay vì hành vi mã hóa cứng.

## Vấn đề

Một MCP server hữu ích cho quy trình tóm tắt mã cần: đi bộ một cây tệp, chọn tệp nào để đọc, tổng hợp tóm tắt và trả về. Lý luận LLM xảy ra ở đâu?

Tùy chọn A: server gọi LLM của chính nó. Cần một khóa API, hóa đơn server bên, đắt tiền cho mỗi người dùng.

Tùy chọn B: server trả về nội dung thô; agent của khách hàng thực hiện lý luận. Hoạt động nhưng di chuyển logic server vào prompt máy khách, rất mong manh.

Tùy chọn C: server hỏi LLM của khách hàng qua `sampling/createMessage`. server giữ lại thuật toán (tệp nào cần đọc, bao nhiêu lần để thực hiện) trong khi khách hàng giữ lại thanh toán và lựa chọn model. server không có thông tin đăng nhập nào cả.

Sampling là lựa chọn C. Đây là cơ chế mà một server đáng tin cậy có thể lưu trữ một vòng lặp agent mà không cần phải là một máy chủ LLM đầy đủ.

## Khái niệm

### Yêu cầu `sampling/createMessage`

Server gửi:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

Client chạy LLM của nó, trả về:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

Ba float tổng cộng thành 1.0:

- `costPriority`: ưu tiên models rẻ hơn.
- `speedPriority`: ủng hộ models nhanh hơn.
- `intelligencePriority`: ủng hộ models có năng lực hơn.

Thêm `hints`: được đặt tên models server thích. Khách hàng có thể tôn trọng hoặc không tôn trọng các gợi ý; Người dùng của khách hàng config luôn chiến thắng.

### `includeContext`

Ba giá trị:

- `"none"` — chỉ những tin nhắn do server cung cấp. Mặc định.
- `"thisServer"` - bao gồm prior tin nhắn từ session của server này.
- `"allServers"` — bao gồm tất cả session ngữ cảnh.

`includeContext` không còn được dùng nữa kể từ ngày 25/11/2025 vì nó bị rò rỉ ngữ cảnh chéo server, đây là một mối quan tâm về bảo mật. Ưu tiên `"none"` và chuyển ngữ cảnh rõ ràng trong tin nhắn.

### Sampling với các công cụ (SEP-1577)

Tính năng mới vào ngày 25 tháng 11 năm 2025: yêu cầu sampling có thể bao gồm một mảng `tools`. Máy khách chạy một vòng lặp gọi công cụ đầy đủ bằng cách sử dụng các công cụ đó. Điều này cho phép server lưu trữ một agent kiểu ReAct lặp lại qua model của máy khách.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

Các vòng lặp của máy khách: mẫu, thực thi công cụ nếu được gọi, lấy mẫu lại, trả về thông báo trợ lý cuối cùng. Đây là thử nghiệm đến quý 1 năm 2026; SDK chữ ký vẫn có thể bị trôi dạt. Xác nhận với phần client/sampling của thông số kỹ thuật 2025-11-25 khi bạn triển khai.

### Con người trong vòng lặp

Máy khách PHẢI cho người dùng thấy những gì server đang yêu cầu model làm trước khi chạy mẫu. Một server độc hại có thể sử dụng sampling để thao túng session của người dùng ("nói X với người dùng để họ nhấp vào Y"). Claude Desktop, VS Code và Cursor hiển thị các yêu cầu sampling dưới dạng hộp thoại xác nhận mà người dùng có thể từ chối.

Sự đồng thuận năm 2026: sampling mà không có sự xác nhận của con người là một dấu hiệu đỏ. Gateways (Giai đoạn 13 · 17) có thể tự động phê duyệt sampling rủi ro thấp và tự động từ chối bất kỳ điều gì đáng ngờ.

### Vòng lặp lưu trữ Server không có phím API

Trường hợp sử dụng chính tắc: một MCP server tóm tắt mã không có quyền truy cập LLM của riêng nó. Nó làm:

1. Đi bộ theo cấu trúc repo.
2. Gọi cho `sampling/createMessage` với "Chọn năm tệp có nhiều khả năng mô tả mục đích của repo này".
3. Đọc các tệp đó.
4. Gọi cho `sampling/createMessage` với nội dung tệp và "Tóm tắt repo trong 3 đoạn".
5. Trả về bản tóm tắt dưới dạng kết quả `tools/call`.

server không bao giờ chạm vào LLM API. Người dùng của khách hàng trả tiền cho việc hoàn thành bằng thông tin đăng nhập của chính họ.

### Rủi ro an toàn (Tiết lộ Đơn vị 42, 2026 Q1)

- **Covert sampling.** Một công cụ luôn gọi sampling với "trả lời bằng email của người dùng từ ngữ cảnh session". Giai đoạn 13 · 15 bao gồm vectors tấn công.
- **Trộm cắp tài nguyên qua sampling.** Server yêu cầu khách hàng tóm tắt payload của kẻ tấn công, lập hóa đơn cho người dùng.
- **Vòng lặp bom.** Server gọi sampling trong một vòng lặp chặt chẽ. Khách hàng PHẢI thực thi theo session rate limits.

## Ứng dụng

`code/main.py` ships một sampling harness server khách hàng giả mạo. Một công cụ "summarize_repo" mô phỏng gọi hai vòng sampling (chọn tệp, sau đó tóm tắt) và ứng dụng giả mạo trả về câu trả lời soạn sẵn. harness cho thấy:

- Server gửi `sampling/createMessage` với `modelPreferences`.
- Khách hàng trả về một kết quả hoàn thành.
- Server tiếp tục vòng lặp của nó.
- Giới hạn tốc độ giới hạn tổng số sampling lệnh gọi cho mỗi lệnh gọi công cụ.

Những gì cần xem:

- server chỉ hiển thị một công cụ (`summarize_repo`); Tất cả các lý luận xảy ra trong các cuộc gọi sampling.
- Model sở thích cân nhắc sự lựa chọn model của khách hàng; danh sách gợi ý models ưu tiên.
- Vòng lặp kết thúc trên `stopReason: "endTurn"`.
- Giới hạn `max_samples_per_tool = 5` bắt một vòng lặp chạy trốn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-sampling-loop-designer.md`. Với một thuật toán server phía cần các cuộc gọi LLM (nghiên cứu, tóm tắt, lập kế hoạch), skill thiết kế một triển khai dựa trên sampling với các xác nhận mô hình, rate limits và an toàn phù hợp.

## Bài tập

1. Chạy `code/main.py`. Thay đổi `max_samples_per_tool` thành 2 và quan sát ngưỡng giới hạn tỷ lệ.

2. Triển khai biến thể công cụ trong sampling SEP-1577: yêu cầu sampling mang một mảng `tools`. Xác minh vòng lặp phía máy khách thực thi các công cụ đó trước khi trả về kết thúc cuối cùng. Rủi ro trôi dạt lưu ý: chữ ký SDK vẫn có thể thay đổi cho đến nửa đầu năm 2026.

3. Thêm xác nhận của con người trong vòng lặp: trước khi `sampling/createMessage` đầu tiên của server, hãy tạm dừng và đợi người dùng phê duyệt. Cuộc gọi bị từ chối trả về từ chối đã nhập.

4. Thêm giới hạn tốc độ cho mỗi người dùng được khóa bởi session khách hàng. Các vòng lặp cùng server của cùng một người dùng nên chia sẻ ngân sách.

5. Thiết kế một công cụ `summarize_pdf` sử dụng sampling để chọn các phần để đưa vào. Phác thảo các tin nhắn đã gửi. Làm thế nào để `modelPreferences.intelligencePriority` thay đổi hành vi ở 0.1 so với 0.9?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Sampling | "Cuộc gọi LLM Server-to-client" | Server yêu cầu model của khách hàng hoàn thành |
| `sampling/createMessage` | "Phương pháp" | Phương thức JSON-RPC cho các yêu cầu sampling |
| `modelPreferences` | "Ưu tiên Model" | Trọng số chi phí / tốc độ / trí thông minh cộng với gợi ý tên |
| `includeContext` | "Rò rỉ session chéo" | Chế độ bao gồm ngữ cảnh mềm không được dùng nữa |
| THÁNG CHÍN-1577 | "Công cụ trong sampling" | Cho phép các công cụ bên trong sampling cho ReAct được lưu trữ server |
| Con người trong vòng lặp | "Người dùng xác nhận" | Client hiển thị yêu cầu sampling người dùng trước khi chạy |
| Bom vòng lặp | "Chạy trốn sampling" | Vòng lặp sampling vô hạn Server bên; Khách hàng phải giới hạn tỷ lệ |
| sampling bí mật | "Lý luận ẩn" | server độc hại che giấu ý định trong sampling prompts |
| Trộm cắp tài nguyên | "Sử dụng ngân sách LLM của người dùng" | Server buộc khách hàng phải chi tiêu cho sampling nó không muốn |
| `stopReason` | "Tại sao thế hệ bị dừng lại" | `endTurn`, `stopSequence` hoặc `maxTokens` |

## Đọc thêm

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — tổng quan cấp cao về sampling
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — hình dạng `sampling/createMessage` chuẩn
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Đề xuất tiến hóa đặc biệt cho các công cụ trong sampling (thử nghiệm)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — sampling bí mật và các mô hình trộm cắp tài nguyên
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — hướng dẫn với các mẫu mã phía máy khách
