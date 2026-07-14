# Tác vụ không đồng bộ (SEP-1686) — Gọi ngay, tìm nạp sau cho công việc dài hạn

> Công việc agent thực tế mất vài phút đến hàng giờ: chạy CI, tổng hợp nghiên cứu sâu batch xuất khẩu. Các lệnh gọi công cụ đồng bộ sẽ ngắt kết nối, hết thời gian chờ hoặc chặn giao diện người dùng. SEP-1686, merged vào ngày 25 tháng 11 năm 2025, bổ sung primitive Nhiệm vụ: bất kỳ yêu cầu nào cũng có thể được tăng cường để trở thành nhiệm vụ và kết quả có thể được tìm nạp sau hoặc phát trực tuyến qua thông báo trạng thái. Lưu ý rủi ro trôi dạt: Các nhiệm vụ đang được thử nghiệm đến nửa đầu năm 2026; SDK bề mặt vẫn đang được thiết kế xung quanh thông số kỹ thuật.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, async task state machine)
**Kiến thức tiên quyết:** Giai đoạn 13 · 07 (MCP server), Giai đoạn 13 · 09 (transports)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Xác định thời điểm cần nâng cấp công cụ từ đồng bộ sang tăng cường tác vụ (>30 giây làm việc bên server).
- Đi theo vòng đời tác vụ: `working` → `input_required` → `completed` / `failed` / `cancelled`.
- Duy trì trạng thái nhiệm vụ để các vụ tai nạn không làm mất công việc trên máy bay.
- Thăm dò ý kiến `tasks/status` và tìm nạp `tasks/result` chính xác.

## Vấn đề

Một công cụ `generate_report` chạy một pipeline trích xuất kéo dài nhiều phút. Các tùy chọn trong model đồng bộ:

1. Giữ kết nối mở trong ba phút. Điều khiển từ xa transports thả nó; khách hàng hết thời gian; Giao diện người dùng bị đóng băng.
2. Quay lại ngay lập tức với một trình giữ chỗ; yêu cầu khách hàng thăm dò ý kiến một endpoint tùy chỉnh. Phá vỡ tính đồng nhất của MCP.
3. Lửa và quên; không có kết quả.

Không có gì là tốt. SEP-1686 bổ sung thứ tư: tăng cường nhiệm vụ. Bất kỳ yêu cầu nào (thường là `tools/call`) đều có thể được gắn thẻ là một nhiệm vụ. server trả về id tác vụ ngay lập tức. Khách hàng thăm dò ý kiến `tasks/status` và tìm nạp `tasks/result` khi hoàn tất. Trạng thái bên Server vẫn tồn tại sau khi khởi động lại.

## Khái niệm

### Tăng cường nhiệm vụ

Một yêu cầu trở thành một nhiệm vụ bằng cách thiết lập `params._meta.task.required: true` (hoặc `optional: true`, server quyết định). server trả lời ngay lập tức với:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` là lời hứa của server về việc giữ lại nhà nước; Sau TTL, kết quả nhiệm vụ sẽ bị loại bỏ.

### Chọn tham gia cho mỗi công cụ

Chú thích công cụ có thể khai báo hỗ trợ tác vụ:

- `taskSupport: "forbidden"` - công cụ này luôn chạy đồng bộ. An toàn cho các công cụ nhanh.
- `taskSupport: "optional"` - khách hàng có thể yêu cầu tăng cường nhiệm vụ.
- `taskSupport: "required"` — máy khách PHẢI sử dụng tính năng tăng cường tác vụ.

Một công cụ `generate_report` sẽ là `required`. Một công cụ `notes_search` sẽ được `forbidden`.

### Tiểu bang

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

Máy trạng thái chỉ nối thêm: một khi `completed`, `failed` hoặc `cancelled`, tác vụ là thiết bị đầu cuối.

### Phương pháp

- `tasks/status {taskId}` — trả về trạng thái hiện tại và gợi ý tiến trình.
- `tasks/result {taskId}` — chặn hoặc trả về 404 nếu chưa hoàn thành.
- `tasks/cancel {taskId}` — idempotent; các quốc gia đầu cuối bỏ qua.
- `tasks/list` - tùy chọn; liệt kê các nhiệm vụ đang hoạt động và đã hoàn thành gần đây.

### Streaming thay đổi trạng thái

Khi server hỗ trợ, máy khách có thể đăng ký nhận thông báo trạng thái:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

Khách hàng phát trực tuyến thay vì thăm dò ý kiến sẽ có trải nghiệm người dùng tốt hơn. Thăm dò ý kiến luôn được hỗ trợ làm bề mặt tối thiểu.

### Trạng thái bền bỉ

Thông số kỹ thuật yêu cầu servers khai báo hỗ trợ tác vụ để duy trì trạng thái. Sự cố không được mất kết quả đã hoàn thành trong ttl. Các cửa hàng bao gồm từ SQLite đến Redis đến hệ thống tệp. Bài 13 harness sử dụng hệ thống tệp.

### Ngữ nghĩa hủy bỏ

`tasks/cancel` là idempotent. Nếu nhiệm vụ đang thực hiện giữa chừng, server cố gắng dừng lại (kiểm tra hủy bỏ thực thi-hợp tác). Nếu đã có thiết bị đầu cuối, yêu cầu là không hoạt động.

### Khôi phục sự cố

Khi khởi động lại server process:

1. Tải tất cả các trạng thái tác vụ liên tục.
2. Đánh dấu bất kỳ nhiệm vụ `working` nào có process chết là `failed` có lỗi `CRASH_RECOVERY`.
3. Bảo tồn `completed` / `failed` / `cancelled` cho ttl của họ.

### Tác vụ không đồng bộ cộng với sampling

Một nhiệm vụ có thể tự gọi `sampling/createMessage`. Đây là cách hoạt động của các nhiệm vụ nghiên cứu kéo dài: nhiệm vụ của server thread lấy mẫu model của khách hàng khi cần thiết, trong khi giao diện người dùng của khách hàng hiển thị nhiệm vụ `working` với các cập nhật tiến độ định kỳ.

### Tại sao đây là thử nghiệm

SEP-1686 shipped vào ngày 25 tháng 11 năm 2025 nhưng lộ trình rộng hơn nêu ra ba vấn đề mở: primitives đăng ký lâu dài, nhiệm vụ con (mối quan hệ nhiệm vụ mẹ-con) và tiêu chuẩn hóa TTL kết quả. Dự kiến thông số kỹ thuật sẽ phát triển đến năm 2026. Mã Production chỉ nên coi Tasks là ổn định đối với trường hợp phổ biến và bảo vệ chống lại các thay đổi SDK trong tương lai đối với các tác vụ con.

## Ứng dụng

`code/main.py` triển khai kho tác vụ bền bỉ (được hỗ trợ bởi hệ thống tệp) và một công cụ `generate_report` chạy trong thread nền. Khách hàng gọi công cụ, nhận id tác vụ ngay lập tức, thăm dò ý kiến `tasks/status` trong khi worker cập nhật tiến độ và tìm nạp `tasks/result` khi hoàn tất. Công việc hủy bỏ; Khôi phục sự cố được mô phỏng bằng cách tắt trạng thái worker thread và tải lại.

Những gì cần xem:

- Trạng thái nhiệm vụ JSON tiếp tục `/tmp/lesson-13-tasks/<id>.json`.
- Worker thread cập nhật `progress` trường; cuộc thăm dò cho thấy nó đang tiến bộ.
- Việc hủy từ phía khách hàng sẽ tạo ra một sự kiện; worker kiểm tra và ra sớm.
- Tải lại trạng thái khi "va chạm" đánh dấu nhiệm vụ trong chuyến bay là `failed` với `CRASH_RECOVERY`.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-task-store-designer.md`. Với một công cụ hoạt động lâu dài (nghiên cứu, xây dựng, xuất), skill thiết kế kho tác vụ (hình dạng trạng thái, ttl, độ bền), chọn cờ taskSupport phù hợp và phác thảo thông báo tiến độ.

## Bài tập

1. Chạy `code/main.py`. Bắt đầu một nhiệm vụ `generate_report`, trạng thái thăm dò ý kiến, sau đó tìm nạp kết quả.

2. Thêm cuộc gọi `tasks/cancel` giữa chừng. Xác minh worker tôn trọng nó và trạng thái trở nên `cancelled`.

3. Mô phỏng khôi phục sự cố: tắt worker thread, khởi động lại bộ nạp và quan sát chế độ lỗi `CRASH_RECOVERY`.

4. Mở rộng cửa hàng sang SQLite. Chiến thắng về độ bền là như nhau; tùy chọn truy vấn mở ra (liệt kê tất cả các tác vụ từ session X).

5. Đọc bài đăng lộ trình MCP cho năm 2026. Xác định một vấn đề mở liên quan đến Nhiệm vụ có nhiều khả năng ảnh hưởng đến thiết kế SDK API trong năm tới.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| Nhiệm vụ | "Lệnh gọi công cụ chạy dài" | Yêu cầu được tăng cường với `_meta.task` để thực thi không đồng bộ |
| THÁNG CHÍN-1686 | "Thông số nhiệm vụ" | Đề xuất phát triển thông số kỹ thuật đã thêm Nhiệm vụ vào 2025-11-25 |
| `_meta.task` | "Phong bì nhiệm vụ" | Siêu dữ liệu cho mỗi yêu cầu chứa id, state, ttl |
| taskHỗ trợ | "Cờ công cụ" | `forbidden` / `optional` / `required` mỗi công cụ |
| `tasks/status` | "Phương pháp thăm dò ý kiến" | Tìm nạp trạng thái hiện tại và gợi ý tiến trình tùy chọn |
| `tasks/result` | "Tìm nạp kết quả" | Trả về payload hoặc 404 đã hoàn thành nếu chưa hoàn thành |
| `tasks/cancel` | "Dừng lại" | Yêu cầu hủy bỏ idempotent |
| TTL | "Ngân sách giữ chân" | Mili giây server hứa hẹn sẽ giữ trạng thái tác vụ |
| `notifications/tasks/updated` | "Đẩy nhà nước" | Sự kiện thay đổi trạng thái do Server khởi tạo |
| Cửa hàng bền bỉ | "Trạng thái an toàn khi va chạm" | Lớp persistence hệ thống tệp / SQLite / Redis |

## Đọc thêm

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — đề xuất ban đầu và thảo luận đầy đủ
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) - Hướng dẫn thiết kế với lý do
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) - cơ khí và máy trạng thái
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — các mẫu thực hiện nhiệm vụ cấp SDK
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — các vấn đề mở và các ưu tiên năm 2026 bao gồm các nhiệm vụ phụ
