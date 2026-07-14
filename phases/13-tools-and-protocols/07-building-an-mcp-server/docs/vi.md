# Xây dựng MCP Server — Python + TypeScript SDKs

> Hầu hết các hướng dẫn MCP chỉ hiển thị các thế giới xin chào. Một server thực sự hiển thị các công cụ cộng với tài nguyên cộng với prompts, xử lý đàm phán năng lực, phát ra các lỗi có cấu trúc và hoạt động tương tự trên SDKs. Bài học này xây dựng một ghi chú server đầu cuối: stdlib stdio transport, JSON-RPC dispatch, ba server primitives và một phong cách chức năng thuần túy rơi vào FastMCP của Python SDK hoặc TypeScript SDK khi bạn tốt nghiệp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, stdio MCP server)
**Kiến thức tiên quyết:** Giai đoạn 13 · 06 (MCP nguyên tắc cơ bản)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Triển khai các phương pháp `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list` và `prompts/get`.
- Viết một vòng lặp điều phối đọc các thông báo JSON-RPC từ stdin và ghi phản hồi cho stdout.
- Phát phản hồi lỗi có cấu trúc theo thông số kỹ thuật JSON-RPC 2.0 và mã bổ sung của MCP.
- Tốt nghiệp triển khai stdlib lên FastMCP (Python SDK) hoặc TypeScript SDK mà không cần viết lại logic công cụ.

## Vấn đề

Trước khi bạn có thể sử dụng transport từ xa (Giai đoạn 13 · 09) hoặc lớp xác thực (Giai đoạn 13 · 16), bạn cần có server cục bộ sạch sẽ. Local có nghĩa là stdio: server được tạo ra bởi máy khách dưới dạng process con, tin nhắn chảy qua stdin/stdout được phân tách bằng dòng mới.

Thông số kỹ thuật 2025-11-25 quy định rằng các thông báo stdio được mã hóa dưới dạng các đối tượng JSON với dấu phân cách `\n` rõ ràng. Không SSE ở đây; SSE là chế độ điều khiển từ xa cũ và sẽ bị loại bỏ vào giữa năm 2026 (Rovo MCP server của Atlassian đã ngừng sử dụng nó vào ngày 30 tháng 6 năm 2026; Keboola vào ngày 1 tháng 4 năm 2026). Đối với stdio, một đối tượng JSON trên mỗi dòng là toàn bộ định dạng dây.

Một server ghi chú là một hình dạng tốt vì nó thực hiện cả ba server primitives. Công cụ làm đột biến (`notes_create`). Tài nguyên hiển thị dữ liệu (`notes://{id}`). Prompts ship bản mẫu (`review_note`). Hình dạng của bài học này khái quát hóa cho bất kỳ lĩnh vực nào.

## Khái niệm

### Vòng lặp công văn

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

Ba quy tắc:

- Không in bất cứ thứ gì không phải là phong bì JSON-RPC. Nhật ký gỡ lỗi chuyển đến stderr.
- Mọi yêu cầu PHẢI được khớp với một phản hồi mang cùng một `id`.
- Thông báo KHÔNG ĐƯỢC trả lời.

### Triển khai `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

Chỉ khai báo những gì bạn ủng hộ. Máy khách dựa vào khả năng được đặt để cổng features.

### Triển khai `tools/list` và `tools/call`

`tools/list` trả về `{tools: [...]}` với mỗi mục nhập có `name`, `description` `inputSchema`. `tools/call` lấy `{name, arguments}` và trả lại `{content: [blocks], isError: bool}`.

Các khối nội dung được nhập. Phổ biến nhất:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

Lỗi công cụ có hai hình dạng. Lỗi cấp giao thức (phương pháp không xác định, tham số xấu) là lỗi JSON-RPC. Lỗi cấp công cụ (lệnh gọi hợp lệ nhưng công cụ không thành công) được trả về dưới dạng `{content: [...], isError: true}`. Điều đó cho phép model nhìn thấy thất bại trong bối cảnh của nó.

### Triển khai tài nguyên

Tài nguyên chỉ đọc theo thiết kế. `resources/list` trả về một bản kê khai; `resources/read` trả về nội dung. URI có thể là `file://...`, `http://...` hoặc lược đồ tùy chỉnh như `notes://`.

Khi bạn hiển thị dữ liệu dưới dạng tài nguyên thay vì công cụ:

- model không "gọi" nó; Khách hàng có thể đưa nó vào ngữ cảnh theo yêu cầu của người dùng.
- Đăng ký cho phép server đẩy các bản cập nhật khi tài nguyên thay đổi (Giai đoạn 13 · 10).
- Giai đoạn 13 · 14 mở rộng điều này với `ui://` cho các tài nguyên tương tác.

### Triển khai prompts

Prompts là các bản mẫu có đối số được đặt tên. Máy chủ hiển thị chúng dưới dạng lệnh gạch chéo. Một `review_note` prompt có thể lấy một đối số `note_id` và tạo ra một mẫu prompt nhiều tin nhắn mà khách hàng cung cấp cho model của nó.

### Sự tinh tế của Stdio transport

- JSON được phân tách bằng dòng mới. Không có khung tiền tố chiều dài.
- Không đệm. `sys.stdout.flush()` sau mỗi lần ghi.
- Khách hàng kiểm soát tuổi thọ. Khi stdin đóng (EOF), hãy thoát sạch sẽ.
- Không xử lý SIGPIPE một cách im lặng; đăng nhập và thoát.

### Chú thích

Mỗi công cụ có thể mang `annotations` mô tả các đặc tính an toàn:

- `readOnlyHint: true` - đọc thuần túy, an toàn để thử lại.
- `destructiveHint: true` - tác dụng phụ không thể đảo ngược; Khách hàng nên xác nhận.
- `idempotentHint: true` - cùng một đầu vào tạo ra cùng một đầu ra.
- `openWorldHint: true` - tương tác với các hệ thống bên ngoài.

Khách hàng sử dụng chúng để quyết định UX (hộp thoại xác nhận, chỉ báo trạng thái) và định tuyến (Giai đoạn 13 · 17).

### Con đường tốt nghiệp

Stdlib server trong `code/main.py` là khoảng 180 dòng. FastMCP (Python) thu gọn logic tương tự thành decorator-style:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK có hình dạng tương đương. Con đường tốt nghiệp là thả vào khi bạn đã sẵn sàng; Các khái niệm (khả năng, điều phối, khối nội dung) đều giống nhau.

## Ứng dụng

`code/main.py` là một ghi chú hoàn chỉnh MCP server trên stdio, stdlib chỉ. Nó xử lý `initialize`, `tools/list` `tools/call` cho ba công cụ (`notes_list`, `notes_search`, `notes_create`), `resources/list` và `resources/read` cho mỗi nốt nhạc và một `review_note` prompt. Bạn có thể điều khiển nó bằng cách gửi thông báo JSON-RPC:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

Những gì cần xem:

- Trình điều phối là một `dict[str, Callable]` được khóa theo tên phương thức.
- Mỗi trình thực thi công cụ trả về một danh sách các khối nội dung, không phải một chuỗi trần.
- `isError: true` được đặt khi trình thực thi tăng lên.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-mcp-server-scaffolder.md`. Được cung cấp một miền (ghi chú, vé, tệp, cơ sở dữ liệu), skill giàn giáo một MCP server với các công cụ / tài nguyên / prompts phù hợp và SDK lộ trình tốt nghiệp.

## Bài tập

1. Chạy `code/main.py` và điều khiển nó bằng các thông báo JSON-RPC được tạo thủ công. Bài tập `notes_create`, sau đó `resources/read` để lấy nốt mới.

2. Thêm một công cụ `notes_delete` với `annotations: {destructiveHint: true}`. Xác minh máy khách sẽ hiển thị hộp thoại xác nhận (điều này yêu cầu một máy chủ thực; Claude Desktop hoạt động).

3. Triển khai `resources/subscribe` để server đẩy `notifications/resources/updated` bất cứ khi nào một ghi chú được sửa đổi. Thêm một nhiệm vụ keepalive.

4. Chuyển server sang FastMCP. Tệp Python sẽ thu nhỏ xuống dưới 80 dòng. Hành vi của dây phải giống hệt nhau; xác minh bằng cùng một harness thử nghiệm JSON-RPC.

5. Đọc phần `server/tools` của thông số kỹ thuật và xác định một trường của định nghĩa công cụ không được triển khai trong server của bài học này. (Gợi ý: có một số; chọn một và thêm nó.)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| MCP server | "Thứ để lộ các công cụ" | Process nói MCP JSON-RPC qua stdio hoặc HTTP |
| transport stdio | "Trẻ em process model" | Server được sinh ra bởi khách hàng; giao tiếp qua stdin/stdout |
| Điều phối viên | "Bộ định tuyến phương pháp" | Ánh xạ tên phương thức JSON-RPC cho hàm xử lý |
| Khối nội dung | "Đoạn kết quả công cụ" | Phần tử được nhập trong mảng `content` của phản hồi công cụ |
| `isError` | "Lỗi cấp công cụ" | Báo hiệu công cụ không thành công; phân biệt với lỗi JSON-RPC |
| Chú thích | "Gợi ý an toàn" | cờ readOnly / phá hủy / idempotent / openWorld |
| FastMCP | "Python SDK" | framework cấp cao hơn dựa trên trình trang trí trên giao thức MCP |
| URI tài nguyên | "Dữ liệu địa chỉ" | `file://`, `db://` hoặc lược đồ tùy chỉnh xác định tài nguyên |
| Mẫu Prompt | "Tóm tắt lệnh gạch chéo" | Mẫu do Server cung cấp với các khe đối số cho giao diện người dùng máy chủ |
| Tuyên bố năng lực | "Feature chuyển đổi" | Cờ trên primitive được khai báo trong `initialize` |

## Đọc thêm

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — tài liệu tham khảo Python triển khai
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — triển khai TS song song
- [FastMCP — server framework](https://gofastmcp.com/) - Python API kiểu trang trí cho MCP servers
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — hướng dẫn từ đầu đến cuối sử dụng một trong hai SDK
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tài liệu tham khảo đầy đủ cho các công cụ/* tin nhắn
