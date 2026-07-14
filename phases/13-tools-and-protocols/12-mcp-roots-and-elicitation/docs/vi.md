# Roots and Elicitation — Phạm vi và đầu vào của người dùng giữa chuyến bay

> Đường dẫn được mã hóa cứng sẽ gặp lỗi khi người dùng mở một dự án khác. Các đối số công cụ được điền sẵn sẽ bị hỏng khi người dùng chỉ định thiếu. Root phạm vi server đến một tập hợp URI do người dùng kiểm soát; elicitation tạm dừng giữa cuộc gọi công cụ để yêu cầu người dùng nhập liệu có cấu trúc thông qua biểu mẫu hoặc URL. Hai primitives máy khách, hai bản sửa lỗi cho các chế độ lỗi MCP phổ biến. SEP-1036 (URL chế độ elicitation, 2025-11-25) đang thử nghiệm đến nửa cuối năm 2026 — hãy kiểm tra SDK phiên bản trước khi tùy thuộc vào nó.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, roots + elicitation demo)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Khai báo `roots` và trả lời `notifications/roots/list_changed`.
- Hạn chế các thao tác tệp server đối với URI bên trong tập gốc đã khai báo.
- Sử dụng `elicitation/create` để yêu cầu người dùng xác nhận hoặc nhập liệu có cấu trúc giữa cuộc gọi công cụ.
- Chọn giữa chế độ biểu mẫu và chế độ URL elicitation (chế độ thứ hai là thử nghiệm; lưu ý rủi ro trôi dạt).

## Vấn đề

Hai thất bại cụ thể một nốt nhạc MCP server thành công trong production.

**Giả định đường dẫn bị hỏng.** server được viết chống lại `~/notes`. Một người dùng trên một máy khác có ghi chú trong `~/Documents/Notes` nhận được một cuộc gọi công cụ không thành công (không tìm thấy tệp) hoặc tệ hơn là viết sai chỗ.

**Thiếu lập luận mà người dùng sẽ biết.** Người dùng yêu cầu "xóa ghi chú báo cáo TPS cũ". model gọi `notes_delete(title: "TPS report")` nhưng có ba ghi chú phù hợp từ năm 2023, 2024 và 2025. Công cụ không thể đoán. Thất bại với "mơ hồ" thật khó chịu; chạy trên cả ba là thảm họa.

Roots sửa lỗi đầu tiên: máy khách khai báo tại `initialize` tập hợp URI mà server có thể chạm vào. Elicitation khắc phục trường hợp thứ hai: server tạm dừng lệnh gọi công cụ và gửi `elicitation/create` yêu cầu người dùng chọn lệnh gọi nào.

## Khái niệm

### Rễ

Máy khách khai báo danh sách gốc tại `initialize`:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

Sau đó, Server có thể gọi cho `roots/list`:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

Servers PHẢI coi gốc là ranh giới: bất kỳ tệp nào đọc hoặc ghi bên ngoài tập gốc đều bị từ chối. Điều này không được thực thi bởi máy khách (server vẫn là mã mà người dùng tin tưởng), nhưng tuân thủ thông số kỹ thuật servers tôn trọng nó.

Khi người dùng thêm hoặc xóa gốc, máy khách sẽ gửi `notifications/roots/list_changed`. server gọi lại `roots/list` và cập nhật ranh giới của nó.

### Tại sao roots là một primitive khách hàng

Root được khai báo bởi máy khách vì chúng đại diện cho sự đồng ý của người dùng model. Người dùng nói với Claude Desktop "cung cấp cho ghi chú này server quyền truy cập vào hai thư mục này". server không thể mở rộng phạm vi đó.

### Elicitation: chế độ biểu mẫu mặc định

`elicitation/create` có dạng schema cộng với prompt ngôn ngữ tự nhiên:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

Client hiển thị một biểu mẫu, thu thập câu trả lời của người dùng, trả về:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

Ba hành động có thể thực hiện: `accept` (người dùng đã điền thông tin đó), `decline` (người dùng đã đóng nó), `cancel` (người dùng hủy toàn bộ lệnh gọi công cụ).

Biểu mẫu schemas phẳng — các đối tượng lồng nhau không được hỗ trợ trong v1. SDKs thường từ chối bất cứ thứ gì phức tạp hơn một lớp.

### Elicitation: Chế độ URL (SEP-1036, thử nghiệm)

Mới vào 2025-11-25. Thay vì schema, server sẽ gửi một URL:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

Client mở URL trong trình duyệt, đợi hoàn tất, trả về khi người dùng quay lại. Hữu ích cho các luồng OAuth, authorization thanh toán và ký tài liệu khi biểu mẫu không đủ.

Lưu ý rủi ro trôi: hình dạng phản hồi SEP-1036 vẫn đang ổn định; một số SDKs trả về URL callback, những người khác trả về token hoàn thành. Đọc ghi chú phát hành của SDK trước khi sử dụng chế độ URL trong production.

### Khi elicitation là công cụ phù hợp

- Xác nhận người dùng trước các hành động phá hoại (gợi ý phá hoại + elicitation).
- Định hướng (chọn một trong N trận đấu).
- Thiết lập lần chạy đầu tiên (API khóa, thư mục, tùy chọn).
- Luồng kiểu OAuth (chế độ URL).

### Khi elicitation sai

- Điền vào các lập luận bắt buộc của một công cụ mà model có thể yêu cầu bằng văn xuôi. Sử dụng prompt lại bình thường, không phải hộp thoại elicitation.
- Cuộc gọi tần suất cao. Elicitation làm gián đoạn cuộc trò chuyện; Không bắn nó bên trong một vòng lặp.
- Bất cứ điều gì mà server có thể xác nhận sau khi thực tế. Xác thực, trả về lỗi, để model hỏi người dùng bằng văn bản.

### Cầu nối giữa con người trong vòng lặp

Elicitation cộng với sampling cùng nhau cho phép model "con người trong vòng lặp" của MCP. Vòng lặp agent của server có thể tạm dừng cho đầu vào của người dùng (elicitation) hoặc suy luận của model (sampling). Giai đoạn 13 · 11 sampling được bảo hiểm; Bài học này bao gồm elicitation. Đặt chúng lại với nhau để kiểm soát vòng giữa đầy đủ.

## Ứng dụng

`code/main.py` mở rộng các ghi chú server với:

- `roots/list` phản hồi rằng server truy vấn lại sau khi thông báo thay đổi danh sách gốc.
- Một công cụ `notes_delete` sử dụng `elicitation/create` để phân biệt khi nhiều nốt khớp nhau.
- Một công cụ `notes_setup` sử dụng elicitation chế độ URL để mở trang config chạy đầu tiên (mô phỏng).
- Kiểm tra ranh giới từ chối các hoạt động trên URI bên ngoài các gốc đã khai báo.

Bản demo chạy ba kịch bản: happy path (một trận đấu), định hướng (ba trận đấu, elicitation cháy), out-of-root-write (bị từ chối).

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-elicitation-form-designer.md`. Cho một công cụ có thể cần xác nhận hoặc định hướng của người dùng, skill thiết kế schema biểu mẫu elicitation và mẫu tin nhắn.

## Bài tập

1. Chạy `code/main.py`. Trigger đường định hướng; Xác nhận câu trả lời của người dùng mô phỏng được chuyển trở lại công cụ.

2. Thêm một `notes_archive` công cụ mới yêu cầu xác nhận elicitation mỗi lần (gợi ý phá hủy). Kiểm tra UX: điều này so với model hỏi lại trong văn bản như thế nào?

3. Triển khai elicitation chế độ URL cho luồng OAuth chạy đầu tiên. Lưu ý rủi ro trôi dạt và thêm bảo vệ phiên bản SDK.

4. Mở rộng `roots/list` xử lý: khi có thông báo, server sẽ đọc lại nguyên tử và quét lại các tay cầm tệp đang mở hiện có thể nằm ngoài phạm vi.

5. Đọc thread thảo luận về vấn đề SEP-1036 trên GitHub. Xác định một câu hỏi mở ảnh hưởng đến cách servers nên xử lý callbacks ở chế độ URL.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Gốc | "Ranh giới đồng ý" | URI mà khách hàng đã cho phép server chạm vào |
| `roots/list` | "Server yêu cầu phạm vi" | Client trả về tập root hiện tại |
| `notifications/roots/list_changed` | "Người dùng thay đổi phạm vi" | Máy khách báo hiệu bộ gốc đã đột biến |
| Elicitation | "Hỏi người dùng giữa cuộc gọi" | Yêu cầu do Server khởi tạo cho đầu vào của người dùng có cấu trúc |
| `elicitation/create` | "Phương pháp" | Phương thức JSON-RPC cho các yêu cầu elicitation |
| Chế độ biểu mẫu | "Hình thức điều khiển Schema" | JSON Schema phẳng được hiển thị dưới dạng biểu mẫu trong giao diện người dùng máy khách |
| Chế độ URL | "Chuyển hướng trình duyệt" | SEP-1036 thử nghiệm; mở một URL và chờ |
| `accept` / `decline` / `cancel` | "Kết quả phản hồi của người dùng" | Ba branches tay cầm server |
| Định hướng | "Chọn một" | Trường hợp sử dụng elicitation phổ biến khi một công cụ có N ứng cử viên |
| Dạng phẳng | "Chỉ tài sản cấp cao nhất" | Elicitation schemas không thể lồng |

## Đọc thêm

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — Tham khảo nguồn gốc chuẩn
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — tài liệu tham khảo elicitation chuẩn
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) - 2025-11-25 Hướng dẫn bổ sung
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — Đề xuất elicitation chế độ URL (thử nghiệm, rủi ro trôi)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — Hướng dẫn UX
